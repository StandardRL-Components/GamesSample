
# Generated: 2025-08-27T20:23:39.984207
# Source Brief: brief_02442.md
# Brief Index: 2442

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    A minimalist side-scrolling platformer where the player uses a single button
    to jump over procedurally generated pits and reach the end goal under a time limit.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # User-facing strings
    user_guide = (
        "Controls: Press space to jump. The player moves forward automatically. "
        "Avoid falling into pits and reach the green flag before the timer runs out."
    )
    game_description = (
        "A fast-paced minimalist platformer. Time your jumps to cross an endless "
        "series of pits. Difficulty increases as you progress. Reach the end to win."
    )

    # Frame advance setting
    auto_advance = True

    # --- Constants ---
    # Screen and World
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    LEVEL_END_X = 15000  # Total length of the level in pixels
    FPS = 30

    # Player
    PLAYER_WIDTH = 28
    PLAYER_HEIGHT = 28
    PLAYER_X_POS = 150  # Fixed X position on the screen
    GRAVITY = 0.9
    JUMP_STRENGTH = -16
    SCROLL_SPEED = 5

    # Level Generation
    PLATFORM_Y = 350
    PLATFORM_HEIGHT = 50
    PLATFORM_SEGMENT_WIDTH = 80
    INITIAL_PIT_GAP_SIZE = 20
    INITIAL_PIT_FREQUENCY = 0.2  # 1 in 5 segments is a pit

    # Colors
    COLOR_BG_TOP = (100, 100, 200)
    COLOR_BG_BOTTOM = (40, 20, 80)
    COLOR_PLAYER = (60, 180, 255)
    COLOR_PLAYER_GLOW = (160, 220, 255)
    COLOR_PLATFORM = (100, 100, 110)
    COLOR_PIT = (10, 5, 20)
    COLOR_FLAG = (0, 220, 100)
    COLOR_TEXT = (255, 255, 255)
    
    # Rewards
    REWARD_WIN = 100.0
    REWARD_FALL = -10.0
    REWARD_TIMEOUT = -10.0
    REWARD_CLEAR_PIT = 1.0
    REWARD_STEP = 0.01  # Small reward for surviving each step


    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Gymnasium spaces
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 32)

        # Pre-render background gradient for performance
        self.background = self._create_gradient_background()
        
        # Game state variables are initialized in reset()
        self.player_pos_y = 0
        self.player_vel_y = 0
        self.on_ground = False
        self.world_scroll = 0
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_left = 0
        self.level_segments = deque()
        self.generated_until_x = 0
        self.cleared_pits = set()
        self.pit_frequency = 0.0
        self.pit_gap_size = 0
        self.particles = []

        # This will call reset() for the first time
        if render_mode == "rgb_array":
            self.reset()
        
        self.validate_implementation()

    def _create_gradient_background(self):
        """Creates a pre-rendered surface with a vertical gradient."""
        bg = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        for y in range(self.SCREEN_HEIGHT):
            # Interpolate color from top to bottom
            interp = y / self.SCREEN_HEIGHT
            color = (
                self.COLOR_BG_TOP[0] * (1 - interp) + self.COLOR_BG_BOTTOM[0] * interp,
                self.COLOR_BG_TOP[1] * (1 - interp) + self.COLOR_BG_BOTTOM[1] * interp,
                self.COLOR_BG_TOP[2] * (1 - interp) + self.COLOR_BG_BOTTOM[2] * interp,
            )
            pygame.draw.line(bg, color, (0, y), (self.SCREEN_WIDTH, y))
        return bg

    def _generate_level_chunks(self):
        """Procedurally generates level segments as needed."""
        while self.generated_until_x < self.world_scroll + self.SCREEN_WIDTH + 200:
            is_pit = self.np_random.random() < self.pit_frequency
            
            if is_pit:
                # Add a pit segment
                pit_width = self.pit_gap_size + self.np_random.integers(0, 21)
                rect = pygame.Rect(self.generated_until_x, self.PLATFORM_Y, pit_width, self.PLATFORM_HEIGHT)
                self.level_segments.append(('pit', rect, self.generated_until_x))
                self.generated_until_x += pit_width
            else:
                # Add a platform segment
                platform_width = self.PLATFORM_SEGMENT_WIDTH + self.np_random.integers(-20, 61)
                rect = pygame.Rect(self.generated_until_x, self.PLATFORM_Y, platform_width, self.PLATFORM_HEIGHT)
                self.level_segments.append(('platform', rect, self.generated_until_x))
                self.generated_until_x += platform_width
        
        # Clean up segments that are far off-screen
        while self.level_segments and self.level_segments[0][1].right < self.world_scroll - 200:
            self.level_segments.popleft()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize game state
        self.player_pos_y = self.PLATFORM_Y - self.PLAYER_HEIGHT
        self.player_vel_y = 0
        self.on_ground = True
        self.world_scroll = 0
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_left = self.FPS * 30  # 30 seconds
        
        # Level state
        self.level_segments.clear()
        self.generated_until_x = 0
        self.cleared_pits.clear()
        self.pit_frequency = self.INITIAL_PIT_FREQUENCY
        self.pit_gap_size = self.INITIAL_PIT_GAP_SIZE
        
        # Create initial safe platform
        start_platform = pygame.Rect(0, self.PLATFORM_Y, self.PLAYER_X_POS + 100, self.PLATFORM_HEIGHT)
        self.level_segments.append(('platform', start_platform, 0))
        self.generated_until_x = start_platform.width
        
        self._generate_level_chunks()

        # Visual effects
        self.particles.clear()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.auto_advance:
            self.clock.tick(self.FPS)

        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Action Handling ---
        space_held = action[1] == 1
        if space_held and self.on_ground:
            self.player_vel_y = self.JUMP_STRENGTH
            self.on_ground = False
            # sfx: jump_sound()

        # --- Game Logic Update ---
        self.steps += 1
        self.time_left -= 1
        reward = self.REWARD_STEP

        # Update difficulty
        if self.steps > 0 and self.steps % 50 == 0:
            self.pit_frequency = min(0.6, self.pit_frequency + 0.02)
            self.pit_gap_size = min(150, self.pit_gap_size + 5)
        
        # Player physics
        self.player_vel_y += self.GRAVITY
        self.player_pos_y += self.player_vel_y
        
        # World scroll
        self.world_scroll += self.SCROLL_SPEED
        
        # Generate new level chunks if needed
        self._generate_level_chunks()
        
        # --- Collision Detection ---
        player_rect = pygame.Rect(self.PLAYER_X_POS, self.player_pos_y, self.PLAYER_WIDTH, self.PLAYER_HEIGHT)
        self.on_ground = False
        
        for seg_type, seg_rect, seg_id in self.level_segments:
            screen_rect = seg_rect.move(-self.world_scroll, 0)
            
            if player_rect.colliderect(screen_rect):
                if seg_type == 'platform':
                    # Check if player is landing on top
                    if self.player_vel_y > 0 and player_rect.bottom < screen_rect.top + self.player_vel_y + 1:
                        self.player_pos_y = screen_rect.top - self.PLAYER_HEIGHT
                        self.player_vel_y = 0
                        if not self.on_ground: # Spawn particles only on first contact
                           self._spawn_landing_particles(player_rect.midbottom)
                           # sfx: land_sound()
                        self.on_ground = True
                elif seg_type == 'pit':
                    self.game_over = True
                    reward += self.REWARD_FALL
                    # sfx: fall_sound()
                    break # Exit collision loop
        
        # Check for falling out of the world
        if self.player_pos_y > self.SCREEN_HEIGHT:
            self.game_over = True
            reward += self.REWARD_FALL
            # sfx: fall_sound()

        # Check for clearing pits
        for seg_type, seg_rect, seg_id in self.level_segments:
            if seg_type == 'pit' and seg_id not in self.cleared_pits:
                if player_rect.left > seg_rect.right - self.world_scroll:
                    self.score += 1
                    reward += self.REWARD_CLEAR_PIT
                    self.cleared_pits.add(seg_id)
                    # sfx: score_point_sound()

        # Update particles
        self._update_particles()

        # --- Termination Conditions ---
        if self.world_scroll + self.PLAYER_X_POS >= self.LEVEL_END_X:
            self.game_over = True
            self.score += 100 # Bonus for finishing
            reward += self.REWARD_WIN
            # sfx: win_sound()
        
        if self.time_left <= 0 and not self.game_over:
            self.game_over = True
            reward += self.REWARD_TIMEOUT
            # sfx: timeout_sound()

        terminated = self.game_over or self.steps >= 1500 # Max episode steps

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _spawn_landing_particles(self, pos):
        for _ in range(10):
            vel = [self.np_random.uniform(-2, 2), self.np_random.uniform(-3, -0.5)]
            size = self.np_random.integers(2, 5)
            lifetime = self.np_random.integers(10, 20)
            self.particles.append([list(pos), vel, size, lifetime])

    def _update_particles(self):
        for p in self.particles:
            p[0][0] += p[1][0] # pos x
            p[0][1] += p[1][1] # pos y
            p[1][1] += 0.2     # gravity on particles
            p[3] -= 1          # lifetime
        self.particles = [p for p in self.particles if p[3] > 0]

    def _render_text(self, text, font, color, pos, shadow=True):
        if shadow:
            text_surface_shadow = font.render(text, True, (0,0,0,100))
            self.screen.blit(text_surface_shadow, (pos[0] + 2, pos[1] + 2))
        text_surface = font.render(text, True, color)
        self.screen.blit(text_surface, pos)

    def _render_game(self):
        # Draw level segments
        for seg_type, seg_rect, _ in self.level_segments:
            screen_rect = seg_rect.move(-self.world_scroll, 0)
            if screen_rect.right > 0 and screen_rect.left < self.SCREEN_WIDTH:
                color = self.COLOR_PLATFORM if seg_type == 'platform' else self.COLOR_PIT
                pygame.draw.rect(self.screen, color, screen_rect)

        # Draw end flag
        flag_x = self.LEVEL_END_X - self.world_scroll
        if flag_x < self.SCREEN_WIDTH:
            pygame.draw.rect(self.screen, self.COLOR_FLAG, (flag_x, self.PLATFORM_Y - 50, 10, 50))
            pygame.draw.polygon(self.screen, self.COLOR_FLAG, [(flag_x+10, self.PLATFORM_Y-50), (flag_x+40, self.PLATFORM_Y-35), (flag_x+10, self.PLATFORM_Y-20)])

        # Draw particles
        for p in self.particles:
            pos, _, size, lifetime = p
            alpha = max(0, min(255, lifetime * 15))
            color = (*self.COLOR_PLAYER_GLOW[:3], alpha)
            s = pygame.Surface((size, size), pygame.SRCALPHA)
            s.fill(color)
            self.screen.blit(s, (int(pos[0]), int(pos[1])))

        # Draw player
        player_rect = pygame.Rect(self.PLAYER_X_POS, int(self.player_pos_y), self.PLAYER_WIDTH, self.PLAYER_HEIGHT)
        
        # Glow effect
        glow_size = int(self.PLAYER_WIDTH * 1.8)
        glow_surface = pygame.Surface((glow_size, glow_size), pygame.SRCALPHA)
        pygame.draw.circle(
            glow_surface,
            (*self.COLOR_PLAYER_GLOW[:3], 80),
            (glow_size // 2, glow_size // 2),
            glow_size // 2
        )
        self.screen.blit(glow_surface, (player_rect.centerx - glow_size // 2, player_rect.centery - glow_size // 2))
        
        # Player body
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=3)

    def _render_ui(self):
        # Score
        score_text = f"SCORE: {self.score}"
        self._render_text(score_text, self.font_small, self.COLOR_TEXT, (10, 10))

        # Timer
        time_display = f"TIME: {max(0, self.time_left / self.FPS):.1f}"
        text_width = self.font_small.size(time_display)[0]
        self._render_text(time_display, self.font_small, self.COLOR_TEXT, (self.SCREEN_WIDTH - text_width - 10, 10))
        
        if self.game_over:
            reason = ""
            if self.world_scroll + self.PLAYER_X_POS >= self.LEVEL_END_X:
                reason = "LEVEL COMPLETE!"
            elif self.time_left <= 0:
                reason = "TIME UP!"
            else:
                reason = "GAME OVER"
            
            self._render_text(reason, self.font_large, self.COLOR_TEXT, (self.SCREEN_WIDTH // 2 - self.font_large.size(reason)[0] // 2, self.SCREEN_HEIGHT // 2 - 50))


    def _get_observation(self):
        # Draw background
        self.screen.blit(self.background, (0, 0))
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_left": self.time_left,
            "world_scroll": self.world_scroll,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Use a display for human play
    pygame.display.set_caption("Minimalist Platformer")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    terminated = False
    running = True
    total_reward = 0
    
    while running:
        # Player input
        keys = pygame.key.get_pressed()
        action = [0, 0, 0] # [movement, space, shift]
        if keys[pygame.K_SPACE]:
            action[1] = 1
        
        # Pygame event loop
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()
                total_reward = 0
                terminated = False

        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            # Render the observation to the display
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()

            if terminated:
                print(f"Episode finished. Total Reward: {total_reward:.2f}, Info: {info}")
        
        # Control frame rate for human play
        env.clock.tick(GameEnv.FPS)

    env.close()