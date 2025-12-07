import gymnasium as gym
import os
import pygame
import pygame.gfxdraw
import numpy as np
import os
import pygame


# Set the SDL video driver to "dummy" to run Pygame headlessly
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from gymnasium.spaces import MultiDiscrete, Box

class GameEnv(gym.Env):
    """
    A puzzle game where the player must make four lights the same color.
    
    The player selects one of four lights, which cycles its color (R->G->B->R).
    This color change propagates to adjacent lights of the same original color,
    creating a chain reaction. The goal is to solve the puzzle in 20 turns or less.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "A puzzle game where the player must make four lights the same color by "
        "triggering chain reactions."
    )
    user_guide = (
        "Use arrow keys to select a light to change its color (↑ top-left, → top-right, ↓ bottom-left, ← bottom-right)."
    )
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 20
        
        # --- Colors ---
        self.COLOR_BG = (15, 18, 28)
        self.COLOR_TEXT = (220, 220, 240)
        self.COLOR_TEXT_SUBTLE = (100, 100, 120)
        self.COLOR_WIN = (100, 255, 100)
        self.COLOR_LOSS = (255, 100, 100)
        self.LIGHT_COLORS = [
            (255, 60, 60),   # Red
            (60, 255, 60),   # Green
            (60, 120, 255)   # Blue
        ]
        self.GLOW_COLORS = [
            (100, 20, 20),
            (20, 100, 20),
            (20, 40, 100)
        ]
        
        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont('Consolas', 48, bold=True)
        self.font_medium = pygame.font.SysFont('Consolas', 24)
        self.font_small = pygame.font.SysFont('Consolas', 16)
        
        # --- Game State ---
        # These are initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.light_states = None
        self.last_action_idx = -1
        self.win_state = False

        # --- Game Layout ---
        self.light_radius = 50
        cx, cy = self.WIDTH // 2, self.HEIGHT // 2 + 20
        offset = 90
        self.light_positions = [
            (cx - offset, cy - offset), (cx + offset, cy - offset),
            (cx - offset, cy + offset), (cx + offset, cy + offset)
        ]
        self.adjacency = {
            0: [1, 2], 1: [0, 3],
            2: [0, 3], 3: [1, 2]
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.last_action_idx = -1 # -1 indicates no action yet
        self.win_state = False

        # Initialize light states, ensuring it's not a solved puzzle
        while True:
            self.light_states = self.np_random.integers(0, 3, size=4, dtype=int)
            if len(set(self.light_states)) > 1:
                break
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]  # 0-4: none/up/down/left/right
        reward = 0.0
        
        # Action mapping: 1-4 correspond to lights 0-3
        action_idx = movement - 1 

        if 0 <= action_idx < 4:
            # A valid light was selected, so a turn is consumed.
            self.steps += 1
            self.last_action_idx = action_idx
            self._propagate_color_change(action_idx)
            reward = self._calculate_reward()
        else:
            # No-op action: do not consume a turn, no state change, no reward.
            self.last_action_idx = -1
        
        self.score += reward
        terminated = self._check_termination()
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # This game does not use truncation
            self._get_info()
        )

    def _propagate_color_change(self, start_index):
        """Handle the core game logic of color propagation."""
        original_color = self.light_states[start_index]
        new_color = (original_color + 1) % 3
        
        q = [start_index]
        visited = {start_index}
        
        while q:
            current_index = q.pop(0)
            # Change the color of the current light
            self.light_states[current_index] = new_color
            
            # Check neighbors
            for neighbor_index in self.adjacency[current_index]:
                if neighbor_index not in visited and self.light_states[neighbor_index] == original_color:
                    visited.add(neighbor_index)
                    q.append(neighbor_index)

    def _calculate_reward(self):
        """Calculate the reward for the current state."""
        is_win = len(set(self.light_states)) == 1
        
        if is_win:
            return 100.0
        
        if self.steps >= self.MAX_STEPS:
            return -100.0
            
        # Continuous and event-based rewards
        reward = 0.0
        num_matching_first = np.sum(self.light_states == self.light_states[0])
        
        # +1 for each light matching the first light's color
        reward += float(num_matching_first)
        
        # +5 bonus for having three lights match
        if num_matching_first == 3:
            reward += 5.0
            
        return reward

    def _check_termination(self):
        """Check if the episode should terminate."""
        is_win = len(set(self.light_states)) == 1
        is_max_steps = self.steps >= self.MAX_STEPS
        
        if is_win:
            self.win_state = True
            self.game_over = True
            return True
        
        if is_max_steps:
            self.game_over = True
            return True
            
        return False

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "turns_left": self.MAX_STEPS - self.steps,
            "is_win": self.win_state
        }

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        """Render the four lights and their glows."""
        for i in range(4):
            pos = self.light_positions[i]
            color_idx = self.light_states[i]
            main_color = self.LIGHT_COLORS[color_idx]
            glow_color = self.GLOW_COLORS[color_idx]
            
            self._draw_glowing_circle(pos, self.light_radius, main_color, glow_color)
            
            # Draw a highlight around the last activated light
            if i == self.last_action_idx:
                pygame.gfxdraw.aacircle(self.screen, int(pos[0]), int(pos[1]), self.light_radius + 8, (255, 255, 255))
                pygame.gfxdraw.aacircle(self.screen, int(pos[0]), int(pos[1]), self.light_radius + 9, (255, 255, 255))

    def _draw_glowing_circle(self, pos, radius, color, glow_color):
        """Draws a circle with a soft outer glow."""
        x, y = int(pos[0]), int(pos[1])
        
        # Draw the glow effect with multiple transparent layers
        for i in range(radius // 2, 0, -2):
            alpha = int(120 * (1 - (i / (radius // 2)))**2)
            glow_surface = pygame.Surface((radius * 2.5, radius * 2.5), pygame.SRCALPHA)
            pygame.draw.circle(glow_surface, (*glow_color, alpha), 
                               (glow_surface.get_width() // 2, glow_surface.get_height() // 2), 
                               radius + i)
            self.screen.blit(glow_surface, (x - glow_surface.get_width() // 2, y - glow_surface.get_height() // 2))

        # Draw the main filled circle with anti-aliasing
        pygame.gfxdraw.filled_circle(self.screen, x, y, radius, color)
        pygame.gfxdraw.aacircle(self.screen, x, y, radius, color)

    def _render_ui(self):
        """Render UI elements like score, turns, and game over messages."""
        # Turns Left
        turns_text = self.font_medium.render(f"Turns Left: {self.MAX_STEPS - self.steps}", True, self.COLOR_TEXT)
        self.screen.blit(turns_text, (20, 20))
        
        # Score
        score_text = self.font_medium.render(f"Score: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 20, 20))
        
        # Title
        title_text = self.font_small.render("Color Chain", True, self.COLOR_TEXT_SUBTLE)
        self.screen.blit(title_text, (self.WIDTH // 2 - title_text.get_width() // 2, 10))

        # Game Over Message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            if self.win_state:
                msg = "VICTORY!"
                color = self.COLOR_WIN
            else:
                msg = "GAME OVER"
                color = self.COLOR_LOSS
                
            end_text = self.font_large.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def close(self):
        pygame.quit()

# Example usage for visualization and testing
if __name__ == '__main__':
    # This block will not run in a headless environment without a display.
    # It's intended for local testing with a GUI.
    try:
        os.environ["SDL_VIDEODRIVER"] = "x11" # Or "windows", "macOS"
        env = GameEnv()
        obs, info = env.reset()
        
        running = True
        pygame_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
        pygame.display.set_caption("Color Chain Gym Environment")
        
        print("--- Manual Control ---")
        print(GameEnv.user_guide)
        print("N: No-op (pass turn)")
        print("R: Reset environment")
        print("Q: Quit")
        
        while running:
            action = [0, 0, 0] # Default to no-op, which does not advance the game
            should_step = False
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    should_step = True
                    if event.key == pygame.K_q:
                        running = False
                        should_step = False
                    elif event.key == pygame.K_r:
                        obs, info = env.reset()
                        print(f"--- Env Reset ---")
                        should_step = False
                    elif event.key == pygame.K_n:
                        action = [0, 0, 0] # No-op
                    elif event.key == pygame.K_UP:
                        action = [1, 0, 0] # Mapped to Light 1 (top-left)
                    elif event.key == pygame.K_RIGHT:
                        action = [2, 0, 0] # Mapped to Light 2 (top-right)
                    elif event.key == pygame.K_DOWN:
                        action = [3, 0, 0] # Mapped to Light 3 (bottom-left)
                    elif event.key == pygame.K_LEFT:
                        action = [4, 0, 0] # Mapped to Light 4 (bottom-right)
                    else:
                        should_step = False # Don't step on other key presses
            
            if should_step:
                obs, reward, terminated, truncated, info = env.step(action)
                print(f"Action: {action[0]}, Reward: {reward:.2f}, Terminated: {terminated}, Info: {info}")
                    
            # Render the environment to the Pygame window
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            pygame_screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            env.clock.tick(env.FPS)
            
        env.close()
    except pygame.error as e:
        print(f"Could not run in graphical mode: {e}")
        print("This is expected in a headless environment.")