
# Generated: 2025-08-27T12:55:35.000675
# Source Brief: brief_00204.md
# Brief Index: 204

        
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
    user_guide = (
        "Controls: ↑ to jump, ←→ to move. Reach the red flag to win."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A procedurally generated platform jumper. Navigate moving platforms to reach the flag within the time limit."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    # Screen
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    
    # Colors
    COLOR_BG_TOP = (20, 20, 80)
    COLOR_BG_BOTTOM = (100, 100, 200)
    COLOR_PLAYER = (50, 255, 50)
    COLOR_PLAYER_GLOW = (150, 255, 150, 64)
    COLOR_PLATFORM = (150, 150, 150)
    COLOR_PLATFORM_OUTLINE = (100, 100, 100)
    COLOR_FLAG = (255, 50, 50)
    COLOR_FLAGPOLE = (200, 200, 200)
    COLOR_UI_TEXT = (255, 255, 255)

    # Physics
    FPS = 30
    GRAVITY = 0.5
    JUMP_STRENGTH = -10
    MOVE_SPEED = 5
    MAX_FALL_SPEED = 12
    FRICTION = 0.85

    # Game Rules
    MAX_LEVELS = 3
    INITIAL_LIVES = 3
    LEVEL_TIME_LIMIT_SECONDS = 60
    LEVEL_TIME_LIMIT_FRAMES = LEVEL_TIME_LIMIT_SECONDS * FPS

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
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_msg = pygame.font.SysFont("monospace", 30, bold=True)

        # State variables are initialized in reset()
        self.player_pos = None
        self.player_vel = None
        self.on_ground = None
        self.platforms = None
        self.flag_rect = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.lives = None
        self.current_level = None
        self.timer = None
        self.highest_platform_y = None
        self.total_score_display = None

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.lives = self.INITIAL_LIVES
        self.current_level = 1
        self.total_score_display = 0
        self.game_over = False
        
        self._reset_level_state()
        
        return self._get_observation(), self._get_info()

    def _reset_level_state(self):
        """Resets the state for the current level (used on level start, or life loss)."""
        self.steps = 0
        self.score = 0
        self.timer = self.LEVEL_TIME_LIMIT_FRAMES
        
        self.player_pos = pygame.Vector2(80, self.SCREEN_HEIGHT - 100)
        self.player_vel = pygame.Vector2(0, 0)
        self.on_ground = False
        self.highest_platform_y = self.SCREEN_HEIGHT
        
        self._generate_platforms()

    def _generate_platforms(self):
        """Procedurally generates platforms for the current level."""
        self.platforms = []
        rng = self.np_random
        
        # Level-based difficulty scaling
        level_params = {
            1: {"v_speed": 0.5, "h_speed": 0.0, "count": 8},
            2: {"v_speed": 1.0, "h_speed": 0.25, "count": 10},
            3: {"v_speed": 1.5, "h_speed": 0.75, "count": 12},
        }
        params = level_params.get(self.current_level, level_params[3])

        # Start platform
        start_platform_rect = pygame.Rect(40, self.SCREEN_HEIGHT - 60, 100, 20)
        self.platforms.append({"rect": start_platform_rect, "move_func": lambda s, r: r})

        # Generate intermediate platforms
        last_x = start_platform_rect.centerx
        last_y = start_platform_rect.top
        
        for i in range(params["count"]):
            w = rng.integers(60, 100)
            h = 20
            
            # Position platforms in a generally upward-right path
            px = last_x + rng.integers(50, 120)
            py = last_y - rng.integers(30, 70)
            
            # Clamp to screen bounds
            px = np.clip(px, w/2, self.SCREEN_WIDTH - w/2)
            py = np.clip(py, h/2, self.SCREEN_HEIGHT - 100)

            rect = pygame.Rect(px - w/2, py - h/2, w, h)
            
            # Movement patterns
            v_amp = rng.uniform(0, 30) * params["v_speed"]
            h_amp = rng.uniform(0, 20) * params["h_speed"]
            freq = rng.uniform(0.01, 0.03)
            phase = rng.uniform(0, 2 * math.pi)

            def make_move_func(base_rect, v_amp, h_amp, freq, phase):
                def move_func(steps, r):
                    new_x = base_rect.x + h_amp * math.sin(freq * steps + phase)
                    new_y = base_rect.y + v_amp * math.cos(freq * steps + phase * 1.5)
                    r.topleft = (new_x, new_y)
                    return r
                return move_func

            self.platforms.append({
                "rect": rect,
                "move_func": make_move_func(rect.copy(), v_amp, h_amp, freq, phase)
            })
            
            last_x = px
            last_y = py

        # Final platform and flag
        final_platform_rect = pygame.Rect(self.SCREEN_WIDTH - 120, 80, 100, 20)
        self.platforms.append({"rect": final_platform_rect, "move_func": lambda s, r: r})
        self.flag_rect = pygame.Rect(final_platform_rect.centerx - 5, final_platform_rect.top - 40, 10, 40)

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Action Handling ---
        movement = action[0]
        player_jumped = (movement == 1)
        player_left = (movement == 3)
        player_right = (movement == 4)
        
        # --- Physics and State Update ---
        # Horizontal movement
        if player_left:
            self.player_vel.x = -self.MOVE_SPEED
        elif player_right:
            self.player_vel.x = self.MOVE_SPEED
        
        # Vertical movement (Jump)
        if player_jumped and self.on_ground:
            self.player_vel.y = self.JUMP_STRENGTH
            self.on_ground = False
            # sfx: jump_sound()

        # Apply gravity
        self.player_vel.y += self.GRAVITY
        self.player_vel.y = min(self.player_vel.y, self.MAX_FALL_SPEED)

        # Apply friction
        self.player_vel.x *= self.FRICTION
        if abs(self.player_vel.x) < 0.1:
            self.player_vel.x = 0

        # Update player position
        self.player_pos += self.player_vel

        # Update platform positions
        for p in self.platforms:
            p["rect"] = p["move_func"](self.steps, p["rect"])

        # --- Collision Detection ---
        player_rect = pygame.Rect(self.player_pos.x, self.player_pos.y, 20, 20)
        self.on_ground = False
        reward = -0.01  # Penalty for being in the air

        for p in self.platforms:
            platform_rect = p["rect"]
            if player_rect.colliderect(platform_rect):
                # Player was above the platform in the previous frame
                if self.player_vel.y > 0 and player_rect.bottom - self.player_vel.y <= platform_rect.top:
                    self.player_pos.y = platform_rect.top - player_rect.height
                    self.player_vel.y = 0
                    self.on_ground = True
                    reward = 0.1 # Reward for being on a platform
                    # sfx: land_sound()
                    
                    # Check for highest platform reward
                    if platform_rect.top < self.highest_platform_y:
                        reward += 5
                        self.highest_platform_y = platform_rect.top
                
                # Player hits platform from below
                elif self.player_vel.y < 0 and player_rect.top - self.player_vel.y >= platform_rect.bottom:
                    self.player_pos.y = platform_rect.bottom
                    self.player_vel.y = 1 # Bounce down
                    reward -= 1 # Penalty for hitting head
                    # sfx: bonk_sound()

                # Side collisions
                else:
                    self.player_vel.x = 0
        
        self.score += reward

        # --- Termination and Level Transition ---
        terminated = False
        is_dead = False
        level_complete = False

        # Check for falling off screen
        if self.player_pos.y > self.SCREEN_HEIGHT:
            is_dead = True
            self.score -= 100
            # sfx: fall_scream()

        # Check for time out
        if self.timer <= 0:
            is_dead = True
            self.score -= 50
            # sfx: timeout_buzzer()

        # Check for reaching the flag
        player_rect.center = self.player_pos
        if player_rect.colliderect(self.flag_rect):
            level_complete = True
            self.score += 100
            # sfx: level_win_jingle()

        # Handle state transitions
        if level_complete:
            self.total_score_display += self.score
            self.current_level += 1
            if self.current_level > self.MAX_LEVELS:
                self.game_over = True
            else:
                self._reset_level_state()
        
        elif is_dead:
            self.total_score_display += self.score
            self.lives -= 1
            if self.lives <= 0:
                self.game_over = True
            else:
                self._reset_level_state()
        
        terminated = self.game_over
        self.steps += 1
        self.timer -= 1
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _get_observation(self):
        self._render_background()
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "total_score": self.total_score_display,
            "steps": self.steps,
            "lives": self.lives,
            "level": self.current_level,
            "timer": self.timer / self.FPS,
        }

    def _render_background(self):
        """Draws a vertical gradient background."""
        for y in range(self.SCREEN_HEIGHT):
            ratio = y / self.SCREEN_HEIGHT
            color = (
                self.COLOR_BG_TOP[0] * (1 - ratio) + self.COLOR_BG_BOTTOM[0] * ratio,
                self.COLOR_BG_TOP[1] * (1 - ratio) + self.COLOR_BG_BOTTOM[1] * ratio,
                self.COLOR_BG_TOP[2] * (1 - ratio) + self.COLOR_BG_BOTTOM[2] * ratio,
            )
            pygame.draw.line(self.screen, color, (0, y), (self.SCREEN_WIDTH, y))

    def _render_game(self):
        # Draw platforms
        for p in self.platforms:
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM, p["rect"])
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM_OUTLINE, p["rect"], 2)

        # Draw flag
        pygame.draw.rect(self.screen, self.COLOR_FLAGPOLE, self.flag_rect)
        flag_points = [
            (self.flag_rect.right, self.flag_rect.top),
            (self.flag_rect.right, self.flag_rect.top + 15),
            (self.flag_rect.right + 20, self.flag_rect.top + 7.5),
        ]
        pygame.draw.polygon(self.screen, self.COLOR_FLAG, flag_points)

        # Draw player
        player_rect = pygame.Rect(self.player_pos.x, self.player_pos.y, 20, 20)
        
        # Glow effect
        glow_surface = pygame.Surface((40, 40), pygame.SRCALPHA)
        pygame.draw.circle(glow_surface, self.COLOR_PLAYER_GLOW, (20, 20), 20)
        self.screen.blit(glow_surface, (player_rect.centerx - 20, player_rect.centery - 20))

        # Player square
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect)
        pygame.draw.rect(self.screen, (255, 255, 255), player_rect, 1)

    def _render_ui(self):
        # Render lives
        lives_text = self.font_ui.render(f"LIVES: {self.lives}", True, self.COLOR_UI_TEXT)
        self.screen.blit(lives_text, (10, 10))

        # Render level
        level_text = self.font_ui.render(f"LEVEL: {self.current_level}/{self.MAX_LEVELS}", True, self.COLOR_UI_TEXT)
        self.screen.blit(level_text, (self.SCREEN_WIDTH / 2 - level_text.get_width() / 2, 10))
        
        # Render timer
        time_left = max(0, self.timer / self.FPS)
        timer_text = self.font_ui.render(f"TIME: {time_left:.1f}", True, self.COLOR_UI_TEXT)
        self.screen.blit(timer_text, (self.SCREEN_WIDTH - timer_text.get_width() - 10, 10))

        # Game over message
        if self.game_over:
            msg = "GAME OVER" if self.lives <= 0 else "YOU WIN!"
            msg_text = self.font_msg.render(msg, True, self.COLOR_UI_TEXT)
            msg_rect = msg_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            pygame.draw.rect(self.screen, (0,0,0,150), msg_rect.inflate(20, 20))
            self.screen.blit(msg_text, msg_rect)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # This block allows you to play the game manually
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Procedural Platformer")
    clock = pygame.time.Clock()
    
    running = True
    while running:
        movement = 0 # no-op
        keys = pygame.key.get_pressed()
        
        if keys[pygame.K_UP]:
            movement = 1
        # K_DOWN is 2, but has no effect
        if keys[pygame.K_LEFT]:
            movement = 3
        if keys[pygame.K_RIGHT]:
            movement = 4

        # The environment uses a MultiDiscrete action space.
        # We only care about the first element for manual play.
        action = [movement, 0, 0]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Draw the observation from the environment to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Info: {info}")
            # Wait a bit before resetting
            pygame.time.wait(3000)
            obs, info = env.reset()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        clock.tick(env.FPS)
        
    env.close()