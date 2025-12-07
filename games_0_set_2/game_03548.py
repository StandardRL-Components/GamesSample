
# Generated: 2025-08-27T23:42:09.821894
# Source Brief: brief_03548.md
# Brief Index: 3548

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = "Controls: Press space to jump."

    # Must be a short, user-facing description of the game:
    game_description = "A side-scrolling arcade game where you control a hopping spaceship, dodging obstacles to reach the end of each stage as quickly as possible."

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    # Screen
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    # World
    WORLD_WIDTH_FACTOR = 4
    STAGE_WIDTH = SCREEN_WIDTH * WORLD_WIDTH_FACTOR
    FLOOR_HEIGHT = 50
    # Player
    PLAYER_WIDTH = 30
    PLAYER_HEIGHT = 20
    PLAYER_START_X = 100
    PLAYER_COLOR = (255, 255, 255)
    PLAYER_GLOW_COLOR = (200, 200, 255)
    # Physics (tuned for 30 FPS)
    TARGET_FPS = 30
    GRAVITY = 0.8
    JUMP_IMPULSE = -14
    BASE_PLAYER_SPEED = 6.0
    # Obstacles
    OBSTACLE_COLOR = (255, 48, 48)
    OBSTACLE_WIDTH = 40
    BASE_OBSTACLE_COUNT = 7
    # Game Flow
    TIME_LIMIT_SECONDS = 30
    MAX_STEPS = TIME_LIMIT_SECONDS * TARGET_FPS
    FAST_FINISH_SECONDS = 15
    # UI
    COLOR_BG = (16, 16, 48)
    COLOR_FLOOR = (64, 64, 80)
    COLOR_TIMER_NORMAL = (48, 255, 48)
    COLOR_TIMER_LOW = (255, 48, 48)
    STAR_COUNT = 100

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 16)
        
        # Internal state variables
        self.stage = 1
        self.player_pos = pygame.Vector2(0, 0)
        self.player_vel_y = 0.0
        self.is_grounded = False
        self.obstacles = []
        self.stars = []
        self.camera_x = 0.0
        self.steps = 0
        self.score = 0.0
        self.timer = 0
        self.game_over = False
        self.win_condition_met = False
        self.last_space_press = False
        
        # Initialize state variables
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # On a true reset (not stage progression), reset stage to 1
        if not getattr(self, 'win_condition_met', False):
            self.stage = 1

        # Game state
        self.steps = 0
        self.score = 0.0
        self.timer = self.MAX_STEPS
        self.game_over = False
        self.win_condition_met = False
        self.last_space_press = False

        # Player state
        self.player_pos = pygame.Vector2(self.PLAYER_START_X, self.SCREEN_HEIGHT - self.FLOOR_HEIGHT - self.PLAYER_HEIGHT)
        self.player_vel_y = 0.0
        self.is_grounded = True

        # World state
        self.camera_x = 0.0
        self._generate_stars()
        self._generate_obstacles()
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            # If the game is over, do nothing until reset, but return valid data
            return self._get_observation(), 0.0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_button = action[1] == 1  # Boolean
        shift_held = action[2] == 1  # Boolean
        
        # Use rising edge detection for jump to prevent holding space for continuous jumps
        space_pressed = space_button and not self.last_space_press
        self.last_space_press = space_button

        # --- Update game logic ---
        # Player jump
        if space_pressed and self.is_grounded:
            self.player_vel_y = self.JUMP_IMPULSE
            self.is_grounded = False
            # sfx: jump_sound()

        # Player physics
        self.player_vel_y += self.GRAVITY
        self.player_pos.y += self.player_vel_y
        
        # Player horizontal movement (constant scrolling)
        player_speed = self.BASE_PLAYER_SPEED + 0.05 * (self.stage - 1)
        self.player_pos.x += player_speed

        # Ground collision
        floor_y = self.SCREEN_HEIGHT - self.FLOOR_HEIGHT
        if self.player_pos.y + self.PLAYER_HEIGHT >= floor_y:
            self.player_pos.y = floor_y - self.PLAYER_HEIGHT
            self.player_vel_y = 0
            if not self.is_grounded:
                # sfx: land_sound()
                self.is_grounded = True
        
        # Ceiling collision
        if self.player_pos.y < 0:
            self.player_pos.y = 0
            self.player_vel_y = 0

        # Update camera to follow player
        self.camera_x = self.player_pos.x - self.PLAYER_START_X
        self.camera_x = max(0, min(self.camera_x, self.STAGE_WIDTH - self.SCREEN_WIDTH))
        
        # Update timer and steps
        self.steps += 1
        self.timer -= 1
        
        # --- Check for Termination and Calculate Reward ---
        reward = 0.1  # Survival reward
        terminated = False
        
        # Check obstacle collision
        player_rect = self._get_player_rect()
        for obstacle in self.obstacles:
            if player_rect.colliderect(obstacle):
                reward = -100.0
                terminated = True
                self.game_over = True
                # sfx: explosion_sound()
                break
        
        if not terminated:
            # Check win condition
            if self.player_pos.x >= self.STAGE_WIDTH - self.PLAYER_START_X:
                self.win_condition_met = True
                reward += 10.0  # Base win reward
                if self.timer > self.MAX_STEPS - (self.FAST_FINISH_SECONDS * self.TARGET_FPS):
                    reward += 50.0 # Fast finish bonus
                terminated = True
                self.game_over = True
                self.stage += 1 # Progress to next stage on next reset
                # sfx: win_sound()

            # Check time out
            elif self.timer <= 0:
                reward = -100.0
                terminated = True
                self.game_over = True
                # sfx: timeout_sound()

        self.score += reward
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )
    
    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "stage": self.stage,
            "timer": self.timer,
        }

    def _get_player_rect(self):
        return pygame.Rect(self.player_pos.x, self.player_pos.y, self.PLAYER_WIDTH, self.PLAYER_HEIGHT)

    def _generate_stars(self):
        self.stars = []
        for _ in range(self.STAR_COUNT * self.WORLD_WIDTH_FACTOR):
            x = self.np_random.integers(0, self.STAGE_WIDTH)
            y = self.np_random.integers(0, self.SCREEN_HEIGHT - self.FLOOR_HEIGHT)
            # parallax_factor: 1=foreground, <1=background
            parallax_factor = self.np_random.uniform(0.1, 0.6)
            size = int(1 + (parallax_factor * 2))
            self.stars.append((x, y, size, parallax_factor))

    def _generate_obstacles(self):
        self.obstacles = []
        num_obstacles = self.BASE_OBSTACLE_COUNT + (self.stage - 1)
        if num_obstacles == 0: return
        
        segment_width = (self.STAGE_WIDTH - 600) // num_obstacles
        
        for i in range(num_obstacles):
            # Place obstacles after an initial safe zone
            start_x = 600 + i * segment_width
            x = self.np_random.integers(start_x, start_x + segment_width - self.OBSTACLE_WIDTH)
            
            # 50/50 chance of being a high or low obstacle
            if self.np_random.random() > 0.5:
                # Low obstacle (jump over)
                height = self.np_random.integers(40, 120)
                y = self.SCREEN_HEIGHT - self.FLOOR_HEIGHT - height
            else:
                # High obstacle (pass under)
                height = self.np_random.integers(150, 250)
                y = 0
            
            self.obstacles.append(pygame.Rect(x, y, self.OBSTACLE_WIDTH, height))

    def _render_game(self):
        # Draw stars with parallax
        for x, y, size, parallax in self.stars:
            star_screen_x = (x - self.camera_x * parallax) % self.SCREEN_WIDTH
            color_val = int(100 + 155 * parallax)
            star_color = (color_val, color_val, color_val)
            self.screen.set_at((int(star_screen_x), int(y)), star_color)
            if size > 1:
                 pygame.draw.circle(self.screen, star_color, (int(star_screen_x), int(y)), size-1)

        # Draw floor
        floor_rect = pygame.Rect(0, self.SCREEN_HEIGHT - self.FLOOR_HEIGHT, self.SCREEN_WIDTH, self.FLOOR_HEIGHT)
        pygame.draw.rect(self.screen, self.COLOR_FLOOR, floor_rect)
        
        # Draw obstacles
        for obstacle in self.obstacles:
            obstacle_screen_pos = obstacle.move(-self.camera_x, 0)
            if obstacle_screen_pos.right > 0 and obstacle_screen_pos.left < self.SCREEN_WIDTH:
                pygame.draw.rect(self.screen, self.OBSTACLE_COLOR, obstacle_screen_pos)
                highlight_rect = obstacle_screen_pos.inflate(-4, -4)
                highlight_color = tuple(min(255, c+40) for c in self.OBSTACLE_COLOR)
                pygame.draw.rect(self.screen, highlight_color, highlight_rect, border_radius=2)

        # Draw player
        player_screen_pos = self.player_pos - pygame.Vector2(self.camera_x, 0)
        
        # Squash and stretch animation for game feel
        squash = max(0, -self.player_vel_y * 0.4) if not self.is_grounded else 0
        stretch = max(0, self.player_vel_y * 0.3) if not self.is_grounded else 0
        
        anim_w = self.PLAYER_WIDTH + squash
        anim_h = self.PLAYER_HEIGHT + stretch
        anim_x = player_screen_pos.x - (anim_w - self.PLAYER_WIDTH) / 2
        anim_y = player_screen_pos.y - (anim_h - self.PLAYER_HEIGHT) / 2
        
        player_draw_rect = pygame.Rect(anim_x, anim_y, anim_w, anim_h)

        # Draw glow effect
        glow_center = (int(player_draw_rect.centerx), int(player_draw_rect.centery))
        for i in range(10, 0, -2):
            alpha = 80 - i * 8
            radius = int(self.PLAYER_WIDTH * 0.7 + i)
            pygame.gfxdraw.filled_circle(
                self.screen, glow_center[0], glow_center[1], radius,
                (*self.PLAYER_GLOW_COLOR, alpha)
            )

        # Draw player polygon (a simple ship shape)
        p1 = (player_draw_rect.centerx, player_draw_rect.top)
        p2 = (player_draw_rect.right, player_draw_rect.bottom)
        p3 = (player_draw_rect.left, player_draw_rect.bottom)
        pygame.draw.polygon(self.screen, self.PLAYER_COLOR, [p1, p2, p3])
        
    def _render_ui(self):
        # Stage text
        stage_text = self.font_large.render(f"STAGE {self.stage}", True, (200, 200, 255))
        self.screen.blit(stage_text, (20, 15))

        # Score text
        score_text = self.font_small.render(f"SCORE: {self.score:.1f}", True, (255, 255, 255))
        self.screen.blit(score_text, (20, 45))

        # Timer text
        time_left_sec = max(0, self.timer / self.TARGET_FPS)
        timer_color = self.COLOR_TIMER_NORMAL if time_left_sec > 5 else self.COLOR_TIMER_LOW
        timer_text = self.font_large.render(f"{time_left_sec:.1f}", True, timer_color)
        self.screen.blit(timer_text, (self.SCREEN_WIDTH - timer_text.get_width() - 20, 15))
        
        # Game over / Win text
        if self.game_over:
            msg = "STAGE CLEAR" if self.win_condition_met else "GAME OVER"
            color = (48, 255, 48) if self.win_condition_met else (255, 48, 48)
            
            end_text = self.font_large.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            
            bg_rect = text_rect.inflate(20, 20)
            s = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
            s.fill((0,0,0,150))
            self.screen.blit(s, bg_rect.topleft)
            
            self.screen.blit(end_text, text_rect)

    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
        '''
        Call this to verify implementation.
        '''
        print("Running implementation validation...")
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), f"Obs shape is {test_obs.shape}"
        assert test_obs.dtype == np.uint8
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        print("✓ Implementation validated successfully")

# Example of how to run the environment for manual play
if __name__ == "__main__":
    env = GameEnv()
    env.validate_implementation()

    render_screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Hopping Spaceship")
    clock = pygame.time.Clock()
    
    obs, info = env.reset()
    terminated = False
    total_reward = 0
    running = True

    while running:
        # Action state
        movement, space_held, shift_held = 0, 0, 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment...")
                obs, info = env.reset()
                total_reward = 0
                terminated = False

        if not terminated:
            keys = pygame.key.get_pressed()
            space_held = 1 if keys[pygame.K_SPACE] else 0
            
            action = [movement, space_held, shift_held]
            obs, reward, term, trunc, info = env.step(action)
            total_reward += reward
            terminated = term

            if terminated:
                print(f"Episode finished! Total Reward: {total_reward:.2f}, Info: {info}")
                if info.get("win_condition_met", False):
                     print("Stage Clear! Press 'R' to start the next stage or quit.")
                else:
                     print("Game Over! Press 'R' to restart or quit.")

        # Render the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        render_screen.blit(surf, (0, 0))
        pygame.display.flip()

        clock.tick(env.TARGET_FPS)

    env.close()