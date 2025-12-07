# Generated: 2025-08-28T03:23:18.670039
# Source Brief: brief_02007.md
# Brief Index: 2007

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: ↑↓←→ to move. Press space to cycle your color."
    )

    game_description = (
        "Navigate a shifting color path to reach the goal in a top-down, color-matching puzzle."
    )

    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    MAX_STEPS = 2000  # Approx 66 seconds

    # Colors
    COLOR_BG_TOP = (15, 20, 35)
    COLOR_BG_BOTTOM = (30, 40, 60)
    PLAYER_COLORS = [
        (255, 80, 80),   # Red
        (80, 255, 80),   # Green
        (80, 150, 255),  # Blue
        (255, 255, 80),  # Yellow
        (255, 80, 255),  # Magenta
    ]
    COLOR_WHITE = (240, 240, 240)
    COLOR_GRAY = (100, 100, 110)

    # Player
    PLAYER_RADIUS = 12
    PLAYER_SPEED = 4

    # Path
    PATH_WIDTH = 100
    SEGMENT_HEIGHT = 30
    PATH_TOTAL_SEGMENTS = 200

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 36)
        self.font_game_over = pygame.font.Font(None, 72)
        
        # State variables are initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_pos = [0, 0]
        self.player_color_index = 0
        self.path_segments = []
        self.scroll_y = 0
        self.scroll_speed = 0
        self.space_was_held = False
        self.particles = []
        self.win_condition_met = False
        
        self.total_path_pixel_length = self.PATH_TOTAL_SEGMENTS * self.SEGMENT_HEIGHT

        self.reset()
        
        # self.validate_implementation() # Commented out for submission

    def _generate_path(self):
        self.path_segments = []
        current_x = self.SCREEN_WIDTH / 2

        # Create a stable starting area to pass the no-op stability test.
        # A stationary agent should survive for a short period.
        num_stable_segments = 5
        stable_color_index = self.np_random.integers(0, len(self.PLAYER_COLORS))

        for i in range(self.PATH_TOTAL_SEGMENTS):
            # Only start the random walk after the stable segments
            if i >= num_stable_segments:
                current_x += self.np_random.integers(-20, 21)
                current_x = np.clip(current_x, self.PATH_WIDTH / 2, self.SCREEN_WIDTH - self.PATH_WIDTH / 2)
            
            y_pos = self.SCREEN_HEIGHT + i * self.SEGMENT_HEIGHT
            rect = pygame.Rect(current_x - self.PATH_WIDTH / 2, y_pos, self.PATH_WIDTH, self.SEGMENT_HEIGHT)
            
            # Use a stable color for the first few segments, then randomize
            color_index = stable_color_index if i < num_stable_segments else self.np_random.integers(0, len(self.PLAYER_COLORS))
            
            self.path_segments.append({
                "rect": rect,
                "color_index": color_index,
            })

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_condition_met = False
        
        self.scroll_y = 0
        self.scroll_speed = 2.0
        
        self.space_was_held = False
        self.particles = []

        self._generate_path()

        # Align player with the generated safe start of the path
        self.player_pos = [self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT - 50]
        self.player_color_index = self.path_segments[0]["color_index"]

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        
        # --- Handle Input & Update Player ---
        reward = self._handle_input(movement, space_held)

        # --- Update Game State ---
        self._update_game_state()

        # --- Check for Collisions & Termination ---
        terminated, collision_penalty = self._check_termination()
        reward += collision_penalty

        # --- Final Reward Calculation ---
        if not terminated:
            reward += 0.01  # Survival reward
        elif self.win_condition_met:
            reward += 100 # Win reward
        
        self.score += reward
        self.steps += 1
        
        truncated = False
        if self.steps >= self.MAX_STEPS:
            terminated = True
            truncated = True # Use truncated for time limit
            
        self.game_over = terminated

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_input(self, movement, space_held):
        # Movement
        if movement == 1: self.player_pos[1] -= self.PLAYER_SPEED  # Up
        elif movement == 2: self.player_pos[1] += self.PLAYER_SPEED  # Down
        elif movement == 3: self.player_pos[0] -= self.PLAYER_SPEED  # Left
        elif movement == 4: self.player_pos[0] += self.PLAYER_SPEED  # Right

        # Clip player to screen bounds
        self.player_pos[0] = np.clip(self.player_pos[0], self.PLAYER_RADIUS, self.SCREEN_WIDTH - self.PLAYER_RADIUS)
        self.player_pos[1] = np.clip(self.player_pos[1], self.PLAYER_RADIUS, self.SCREEN_HEIGHT - self.PLAYER_RADIUS)

        # Color Cycling
        color_change_reward = 0
        if space_held and not self.space_was_held:
            # Sound effect placeholder: # SFX: Color_Switch
            old_color = self.PLAYER_COLORS[self.player_color_index]
            self.player_color_index = (self.player_color_index + 1) % len(self.PLAYER_COLORS)
            
            # Add particle effect for color change
            self._add_particle(self.player_pos, old_color, 20)
            
            # Reward for switching to correct color
            on_segment, segment = self._get_current_segment()
            if on_segment and self.player_color_index == segment["color_index"]:
                color_change_reward = 1.0

        self.space_was_held = space_held
        return color_change_reward
    
    def _update_game_state(self):
        # Increase difficulty
        if self.steps > 0 and self.steps % 500 == 0:
            self.scroll_speed += 0.05
        
        # Scroll path
        self.scroll_y += self.scroll_speed
        for segment in self.path_segments:
            segment["rect"].y -= self.scroll_speed
            
        # Update particles
        self.particles = [p for p in self.particles if p["life"] > 0]
        for p in self.particles:
            p["radius"] += p["expand_rate"]
            p["life"] -= 1

    def _get_current_segment(self):
        player_x, player_y = self.player_pos
        for segment in self.path_segments:
            if segment["rect"].collidepoint(player_x, player_y):
                return True, segment
        return False, None

    def _check_termination(self):
        # 1. Win Condition: Player has survived the entire path length
        if self.scroll_y >= self.total_path_pixel_length:
            self.win_condition_met = True
            return True, 0

        # 2. Loss Condition: Collision with wrong color
        on_segment, segment = self._get_current_segment()
        if on_segment:
            if self.player_color_index != segment["color_index"]:
                # Sound effect placeholder: # SFX: Mismatch_Fail
                self._add_particle(self.player_pos, (255,255,255), 40, 30)
                return True, -10 # Terminated, penalty
        
        # 3. Loss Condition: Falling off the path (into the void)
        elif self.player_pos[1] > self.SCREEN_HEIGHT - self.PLAYER_RADIUS * 2: # Check only near bottom
            is_safe = False
            for seg in self.path_segments:
                if seg["rect"].top < self.SCREEN_HEIGHT:
                    is_safe = True
                    break
            if not is_safe: # No path segments left on screen
                 return True, -10

        return False, 0 # Not terminated, no penalty

    def _add_particle(self, pos, color, max_radius, life=20, expand_rate=1):
        self.particles.append({
            "pos": list(pos),
            "color": color,
            "radius": 5,
            "max_radius": max_radius,
            "life": life,
            "max_life": life,
            "expand_rate": expand_rate
        })

    def _get_observation(self):
        self._render_background()
        self._render_game()
        self._render_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        # Draw a vertical gradient
        for y in range(self.SCREEN_HEIGHT):
            interp = y / self.SCREEN_HEIGHT
            color = (
                self.COLOR_BG_TOP[0] * (1 - interp) + self.COLOR_BG_BOTTOM[0] * interp,
                self.COLOR_BG_TOP[1] * (1 - interp) + self.COLOR_BG_BOTTOM[1] * interp,
                self.COLOR_BG_TOP[2] * (1 - interp) + self.COLOR_BG_BOTTOM[2] * interp,
            )
            pygame.draw.line(self.screen, color, (0, y), (self.SCREEN_WIDTH, y))

    def _render_game(self):
        # Render path
        for segment in self.path_segments:
            if -self.SEGMENT_HEIGHT < segment["rect"].y < self.SCREEN_HEIGHT:
                color = self.PLAYER_COLORS[segment["color_index"]]
                pygame.draw.rect(self.screen, color, segment["rect"])

        # Render particles
        for p in self.particles:
            alpha = int(255 * (p["life"] / p["max_life"]))
            if alpha > 0:
                pos = (int(p["pos"][0]), int(p["pos"][1]))
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], int(p["radius"]), (*p["color"], alpha))

        # Render player aura/glow
        player_color = self.PLAYER_COLORS[self.player_color_index]
        player_int_pos = (int(self.player_pos[0]), int(self.player_pos[1]))
        for i in range(10):
            alpha = 80 - i * 8
            radius = self.PLAYER_RADIUS + i * 1.5
            pygame.gfxdraw.aacircle(self.screen, player_int_pos[0], player_int_pos[1], int(radius), (*player_color, alpha))
            pygame.gfxdraw.filled_circle(self.screen, player_int_pos[0], player_int_pos[1], int(radius), (*player_color, alpha))

        # Render player core
        pygame.gfxdraw.aacircle(self.screen, player_int_pos[0], player_int_pos[1], self.PLAYER_RADIUS, self.COLOR_WHITE)
        pygame.gfxdraw.filled_circle(self.screen, player_int_pos[0], player_int_pos[1], self.PLAYER_RADIUS, self.COLOR_WHITE)

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"Score: {int(self.score)}", True, self.COLOR_WHITE)
        self.screen.blit(score_text, (10, 10))

        # Time
        time_left = (self.MAX_STEPS - self.steps) / self.FPS
        time_text = self.font_ui.render(f"Time: {max(0, time_left):.1f}", True, self.COLOR_WHITE)
        time_rect = time_text.get_rect(topright=(self.SCREEN_WIDTH - 10, 10))
        self.screen.blit(time_text, time_rect)
        
        # Game Over / Win message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            message = "YOU WIN!" if self.win_condition_met else "GAME OVER"
            color = self.PLAYER_COLORS[1] if self.win_condition_met else self.PLAYER_COLORS[0]
            
            end_text = self.font_game_over.render(message, True, color)
            end_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, end_rect)


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_pos": tuple(self.player_pos),
            "scroll_y": self.scroll_y
        }

    def close(self):
        pygame.font.quit()
        pygame.quit()

    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation.
        """
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    # It is not used by the evaluation environment.
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup Pygame window for human play
    pygame.display.set_caption("Color Path")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    terminated = False
    running = True
    
    while running:
        # --- Human Input ---
        movement = 0 # No-op
        space_held = 0
        shift_held = 0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
        
        action = [movement, space_held, shift_held]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()
                terminated = False

        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Rendering ---
        # The observation is already a rendered frame, so we just need to display it.
        # Pygame uses (width, height), numpy uses (height, width), so we need to transpose.
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(GameEnv.FPS)

    env.close()