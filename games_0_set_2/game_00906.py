
# Generated: 2025-08-27T15:09:51.727927
# Source Brief: brief_00906.md
# Brief Index: 906

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    A top-down arcade game where the player controls a square, dodging randomly
    moving circles to survive as long as possible. The circles accelerate over time.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # User-facing strings
    user_guide = (
        "Controls: Arrow keys to move the white square. Survive for 60 seconds."
    )
    game_description = (
        "A fast-paced arcade game. Dodge the red circles to survive as long as "
        "possible. The circles get faster over time!"
    )

    # Frame advance setting
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 60

    # Player settings
    PLAYER_SIZE = 20
    PLAYER_SPEED = 5
    PLAYER_COLOR = (255, 255, 255)
    PLAYER_GLOW_COLOR = (200, 200, 255)

    # Enemy settings
    NUM_CIRCLES = 10
    CIRCLE_RADIUS = 10
    INITIAL_CIRCLE_MAX_SPEED = 2.0
    ENEMY_COLOR = (255, 50, 50)
    ENEMY_GLOW_COLOR = (180, 40, 40)

    # Game flow settings
    MAX_STEPS = 3600  # 60 seconds at 60 FPS
    DIFFICULTY_INTERVAL = 600 # Increase speed every 10 seconds
    DIFFICULTY_INCREASE = 0.2 # Multiplier increase

    # UI settings
    BG_COLOR = (15, 15, 25)
    UI_TEXT_COLOR = (240, 240, 240)
    UI_PANEL_COLOR = (30, 30, 40, 180) # Semi-transparent
    WIN_COLOR = (100, 255, 100)
    LOSE_COLOR = (255, 100, 100)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        
        # Fonts
        self.font_ui = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_gameover = pygame.font.SysFont("monospace", 48, bold=True)
        
        # Game state variables (initialized in reset)
        self.player_pos = [0, 0]
        self.circles = []
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.win = False
        self.speed_multiplier = 1.0
        self.np_random = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize player
        self.player_pos = [self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2]
        
        # Initialize enemies
        self.circles = []
        for _ in range(self.NUM_CIRCLES):
            while True:
                pos = [
                    self.np_random.integers(self.CIRCLE_RADIUS, self.SCREEN_WIDTH - self.CIRCLE_RADIUS),
                    self.np_random.integers(self.CIRCLE_RADIUS, self.SCREEN_HEIGHT - self.CIRCLE_RADIUS)
                ]
                # Ensure circles don't spawn too close to the player
                if math.dist(pos, self.player_pos) > self.PLAYER_SIZE * 3:
                    break
            
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1.0, self.INITIAL_CIRCLE_MAX_SPEED)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            
            self.circles.append({"pos": pos, "vel": vel})
            
        # Reset game state
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.win = False
        self.speed_multiplier = 1.0
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            # If the game is over, subsequent steps do nothing but return the final state
            return self._get_observation(), 0.0, True, False, self._get_info()

        movement = action[0]  # 0-4: none/up/down/left/right
        
        # --- Update Game Logic ---
        self._update_player(movement)
        self._update_circles()
        self._check_collisions()
        self._update_difficulty()

        self.steps += 1

        # Check for termination conditions
        terminated = self.game_over or self.steps >= self.MAX_STEPS
        if terminated and not self.game_over:
            self.win = True
        self.game_over = terminated

        # --- Calculate Reward ---
        reward = self._calculate_reward(movement, terminated)
        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_player(self, movement):
        if movement == 1: # Up
            self.player_pos[1] -= self.PLAYER_SPEED
        elif movement == 2: # Down
            self.player_pos[1] += self.PLAYER_SPEED
        elif movement == 3: # Left
            self.player_pos[0] -= self.PLAYER_SPEED
        elif movement == 4: # Right
            self.player_pos[0] += self.PLAYER_SPEED
        
        # Clamp player position to screen bounds
        half_size = self.PLAYER_SIZE / 2
        self.player_pos[0] = max(half_size, min(self.player_pos[0], self.SCREEN_WIDTH - half_size))
        self.player_pos[1] = max(half_size, min(self.player_pos[1], self.SCREEN_HEIGHT - half_size))

    def _update_circles(self):
        for circle in self.circles:
            circle['pos'][0] += circle['vel'][0] * self.speed_multiplier
            circle['pos'][1] += circle['vel'][1] * self.speed_multiplier
            
            # Wall bouncing logic
            if not (self.CIRCLE_RADIUS <= circle['pos'][0] <= self.SCREEN_WIDTH - self.CIRCLE_RADIUS):
                circle['vel'][0] *= -1
                circle['pos'][0] = max(self.CIRCLE_RADIUS, min(circle['pos'][0], self.SCREEN_WIDTH - self.CIRCLE_RADIUS))
            if not (self.CIRCLE_RADIUS <= circle['pos'][1] <= self.SCREEN_HEIGHT - self.CIRCLE_RADIUS):
                circle['vel'][1] *= -1
                circle['pos'][1] = max(self.CIRCLE_RADIUS, min(circle['pos'][1], self.SCREEN_HEIGHT - self.CIRCLE_RADIUS))

    def _check_collisions(self):
        player_rect = pygame.Rect(
            self.player_pos[0] - self.PLAYER_SIZE / 2,
            self.player_pos[1] - self.PLAYER_SIZE / 2,
            self.PLAYER_SIZE,
            self.PLAYER_SIZE
        )
        for circle in self.circles:
            if self._rect_circle_collision(player_rect, circle['pos'], self.CIRCLE_RADIUS):
                self.game_over = True
                # sfx: player_hit.wav
                break

    def _rect_circle_collision(self, rect, circle_pos, circle_radius):
        closest_x = max(rect.left, min(circle_pos[0], rect.right))
        closest_y = max(rect.top, min(circle_pos[1], rect.bottom))
        dist_x = circle_pos[0] - closest_x
        dist_y = circle_pos[1] - closest_y
        return (dist_x**2 + dist_y**2) < (circle_radius**2)

    def _update_difficulty(self):
        if self.steps > 0 and self.steps % self.DIFFICULTY_INTERVAL == 0:
            self.speed_multiplier += self.DIFFICULTY_INCREASE
            # sfx: level_up.wav

    def _calculate_reward(self, movement, terminated):
        if terminated:
            return 100.0 if self.win else -10.0
        
        reward = 0.1  # Survival reward
        if movement == 0:
            reward -= 0.2  # Penalty for inaction
        return reward

    def _get_observation(self):
        self.clock.tick(self.FPS)
        self.screen.fill(self.BG_COLOR)
        
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render circles with a glow effect
        for circle in self.circles:
            pos = (int(circle['pos'][0]), int(circle['pos'][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.CIRCLE_RADIUS + 3, self.ENEMY_GLOW_COLOR)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.CIRCLE_RADIUS, self.ENEMY_COLOR)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.CIRCLE_RADIUS, self.ENEMY_COLOR)

        # Render player with a glow effect
        player_rect = pygame.Rect(0, 0, self.PLAYER_SIZE, self.PLAYER_SIZE)
        player_rect.center = (int(self.player_pos[0]), int(self.player_pos[1]))
        
        glow_rect = player_rect.inflate(6, 6)
        glow_surface = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(glow_surface, (*self.PLAYER_GLOW_COLOR, 60), glow_surface.get_rect(), border_radius=5)
        self.screen.blit(glow_surface, glow_rect.topleft)
        
        pygame.draw.rect(self.screen, self.PLAYER_COLOR, player_rect, border_radius=3)

    def _render_ui(self):
        # UI Panel
        ui_panel = pygame.Surface((200, 50), pygame.SRCALPHA)
        ui_panel.fill(self.UI_PANEL_COLOR)
        self.screen.blit(ui_panel, (10, 10))

        # Timer
        time_left = max(0, (self.MAX_STEPS - self.steps) / self.FPS)
        timer_text = f"TIME: {time_left:.1f}"
        self._render_text(timer_text, self.font_ui, self.UI_TEXT_COLOR, (110, 35))

        # Game Over/Win message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 170))
            self.screen.blit(overlay, (0, 0))
            
            message = "YOU WIN!" if self.win else "GAME OVER"
            color = self.WIN_COLOR if self.win else self.LOSE_COLOR
            self._render_text(message, self.font_gameover, color, (self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            
    def _render_text(self, text, font, color, center_pos):
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect(center=center_pos)
        self.screen.blit(text_surface, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "win": self.win,
            "speed_multiplier": self.speed_multiplier,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """Call this to verify the implementation adheres to the Gymnasium API."""
        print("Running implementation validation...")
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test reset and observation space
        obs, info = self.reset(seed=123)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), f"Obs shape is {obs.shape}"
        assert obs.dtype == np.uint8
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")