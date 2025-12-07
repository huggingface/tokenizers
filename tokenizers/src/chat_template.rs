use chrono::Local;
use minijinja::{Environment, ErrorKind, Template};
use minijinja_contrib::pycompat;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Custom Jinja2 error type for chat template rendering
#[derive(Error, Debug)]
#[error("Chat template error: {0}")]
pub struct ChatTemplateError(String);

/// Chat message role (system, user, assistant)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Message {
    pub role: String,
    pub content: String,
}

impl Message {
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
        }
    }
}

/// Inputs for chat template rendering
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatTemplateInputs {
    pub messages: Vec<Message>,
    pub add_generation_prompt: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bos_token: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub eos_token: Option<String>,
}

impl ChatTemplateInputs {
    pub fn new(messages: Vec<Message>, add_generation_prompt: bool) -> Self {
        Self {
            messages,
            add_generation_prompt,
            bos_token: None,
            eos_token: None,
        }
    }

    pub fn with_special_tokens(
        mut self,
        bos_token: Option<String>,
        eos_token: Option<String>,
    ) -> Self {
        self.bos_token = bos_token;
        self.eos_token = eos_token;
        self
    }
}

/// Raise a exception (custom function) used in the chat templates
pub(crate) fn raise_exception(err_text: String) -> Result<String, minijinja::Error> {
    Err(minijinja::Error::new(ErrorKind::SyntaxError, err_text))
}

/// Get the current date in a specific format (custom function), similar to `datetime.now().strftime()` in Python
pub(crate) fn strftime_now(format_str: String) -> Result<String, minijinja::Error> {
    Ok(Local::now().format(&format_str).to_string())
}

/// Compiled chat template for rendering messages
#[derive(Debug, Clone)]
pub struct ChatTemplate {
    template: Template<'static, 'static>,
    bos_token: Option<String>,
    eos_token: Option<String>,
}

impl ChatTemplate {
    /// Create a new chat template from a template string
    pub fn new(
        template: String,
        bos_token: Option<String>,
        eos_token: Option<String>,
    ) -> Result<Self, ChatTemplateError> {
        let mut env = Box::new(Environment::new());
        // enable things like .strip() or .capitalize()
        env.set_unknown_method_callback(pycompat::unknown_method_callback);

        // Apply template mutations for compatibility
        let mutated_template = template
            // Hack to adjust gemma3 template for debug
            // replace 'messages[0]['content'][0]['text']' with 'messages[0]['content']'
            .replace("messages[0]['content'][0]['text']", "messages[0]['content']")
            // Hack to fix Qwen3 templating - reverse list notation
            .replace("[::-1]", "|reverse")
            // Hack to remove generation markers from training templates
            .replace("{% generation %}", "")
            .replace("{% endgeneration %}", "");

        let template_str = mutated_template.into_boxed_str();
        env.add_function("raise_exception", raise_exception);
        env.add_function("strftime_now", strftime_now);

        // Leak env and template_str as read-only, static resources for performance
        let template = Box::leak(env)
            .template_from_str(Box::leak(template_str))
            .map_err(|e| ChatTemplateError(format!("Failed to compile template: {}", e)))?;

        Ok(Self {
            template,
            bos_token,
            eos_token,
        })
    }

    /// Apply the chat template to messages
    pub fn apply(
        &self,
        mut inputs: ChatTemplateInputs,
    ) -> Result<String, ChatTemplateError> {
        // Add special tokens to inputs if available
        if self.bos_token.is_some() {
            inputs.bos_token = self.bos_token.clone();
        }
        if self.eos_token.is_some() {
            inputs.eos_token = self.eos_token.clone();
        }

        // Render template
        self.template
            .render(&inputs)
            .map_err(|e| ChatTemplateError(format!("Template rendering failed: {}", e)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_chat_template() {
        let template_str = r#"
        {% for message in messages %}
            {% if message['role'] == 'user' %}
                User: {{ message['content'] }}
            {% elif message['role'] == 'assistant' %}
                Assistant: {{ message['content'] }}
            {% endif %}
        {% endfor %}
        "#;

        let ct = ChatTemplate::new(template_str.to_string(), None, None)
            .expect("Failed to create template");

        let messages = vec![
            Message::new("user", "Hello"),
            Message::new("assistant", "Hi there!"),
        ];

        let inputs = ChatTemplateInputs::new(messages, false);
        let result = ct.apply(inputs).expect("Failed to apply template");

        assert!(result.contains("User: Hello"));
        assert!(result.contains("Assistant: Hi there!"));
    }

    #[test]
    fn test_template_with_special_tokens() {
        let template_str = r#"{{ bos_token }}{% for message in messages %}[{{ message['role'] }}]: {{ message['content'] }}
{% endfor %}{{ eos_token }}"#;

        let ct = ChatTemplate::new(
            template_str.to_string(),
            Some("<bos>".to_string()),
            Some("<eos>".to_string()),
        )
        .expect("Failed to create template");

        let messages = vec![Message::new("user", "Hello")];
        let inputs = ChatTemplateInputs::new(messages, false);
        let result = ct.apply(inputs).expect("Failed to apply template");

        assert!(result.starts_with("<bos>"));
        assert!(result.ends_with("<eos>"));
    }

    #[test]
    fn test_template_with_add_generation_prompt() {
        let template_str = r#"{% for message in messages %}{{ message['content'] }}
{% endfor %}{% if add_generation_prompt %}Assistant: {% endif %}"#;

        let ct = ChatTemplate::new(template_str.to_string(), None, None)
            .expect("Failed to create template");

        let messages = vec![Message::new("user", "Hello")];
        let inputs = ChatTemplateInputs::new(messages, true);
        let result = ct.apply(inputs).expect("Failed to apply template");

        assert!(result.contains("Assistant:"));
    }

    #[test]
    fn test_template_with_raise_exception() {
        let mut env = Environment::new();
        env.add_function("raise_exception", raise_exception);

        let template_str = r#"{% if messages|length == 0 %}{{ raise_exception("No messages provided") }}{% endif %}"#;
        let tmpl = env
            .template_from_str(template_str)
            .expect("Failed to compile template");

        let inputs = serde_json::json!({
            "messages": [],
            "add_generation_prompt": false
        });

        let result = tmpl.render(inputs);
        assert!(result.is_err());
    }

    #[test]
    fn test_template_with_strftime() {
        let template_str = r#"{% set today = strftime_now("%Y-%m-%d") %}Date: {{ today }}"#;

        let ct = ChatTemplate::new(template_str.to_string(), None, None)
            .expect("Failed to create template");

        let messages = vec![];
        let inputs = ChatTemplateInputs::new(messages, false);
        let result = ct.apply(inputs).expect("Failed to apply template");

        assert!(result.contains("Date:"));
        // Should contain a date like "2025-12-07"
        assert!(result.len() > 10);
    }

    #[test]
    fn test_message_creation() {
        let msg = Message::new("user", "Hello");
        assert_eq!(msg.role, "user");
        assert_eq!(msg.content, "Hello");
    }
}
