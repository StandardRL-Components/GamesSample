
# Generated: 2025-08-28T06:42:09.367720
# Source Brief: brief_03005.md
# Brief Index: 3005

        
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
        "Controls: Press Space to jump to the beat. Time your jumps to land on the platforms."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A side-scrolling rhythm game. Jump to the beat to traverse platforms and reach the end before time runs out. Perfect timing is key!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        self.WIDTH, self.HEIGHT = 640, 400
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        
        # Game Constants
        self.FPS = 30
        self.BPM = 180
        self.FRAMES_PER_BEAT = (60 / self.BPM) * self.FPS
        self.LEVEL_LENGTH = 12000  # pixels
        self.SCROLL_SPEED = self.LEVEL_LENGTH / (60 * self.FPS)

        # Player Constants
        self.GRAVITY = 0.8
        self.JUMP_STRENGTH = -15
        self.PLAYER_SIZE = (20, 20)
        self.PLAYER_X_POS = 100

        # Colors
        self.COLOR_BG_START = (10, 5, 20)
        self.COLOR_BG_END = (30, 10, 50)
        self.COLOR_PLAYER = (255, 255, 0)
        self.COLOR_PLATFORM_CYCLE = [(255, 0, 128), (0, 255, 255), (128, 0, 255)]
        self.COLOR_FINISH = (255, 255, 255)
        self.COLOR_UI = (220, 220, 220)
        self.COLOR_PULSE = (255, 255, 255)
        self.COLOR_PERFECT = (0, 255, 128)
        self.COLOR_GOOD = (255, 165, 0)

        # Fonts
        self.font_ui = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_feedback = pygame.font.SysFont("Consolas", 18, bold=True)

        # State variables (initialized in reset)
        self.player_pos = [0, 0]
        self.player_vel_y = 0
        self.on_ground = False
        self.prev_space_held = False
        self.frame_count = 0
        self.score = 0
        self.lives = 3
        self.game_over = False
        self.timer = 60.0
        self.world_offset_x = 0
        self.platforms = []
        self.last_safe_platform_idx = 0
        self.feedback_messages = []
        self.particles = []
        
        self.reset()
        self.validate_implementation()

    def _generate_platforms(self):
        platforms = []
        # Start platform
        platforms.append(pygame.Rect(self.PLAYER_X_POS - 40, self.HEIGHT - 100, 120, 20))
        
        current_x = platforms[0].right
        current_y = platforms[0].y
        
        while current_x < self.LEVEL_LENGTH + self.WIDTH:
            gap = self.np_random.integers(100, 180)
            y_change = self.np_random.integers(-60, 60)
            width = self.np_random.integers(80, 150)
            
            current_x += gap
            new_y = np.clip(current_y + y_change, 100, self.HEIGHT - 50)
            
            platforms.append(pygame.Rect(current_x, new_y, width, 20))
            current_x += width
            current_y = new_y
            
        return platforms

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.platforms = self._generate_platforms()
        
        start_platform = self.platforms[0]
        self.player_pos = [self.PLAYER_X_POS, start_platform.top - self.PLAYER_SIZE[1]]
        self.player_vel_y = 0
        self.on_ground = True
        self.prev_space_held = False
        
        self.frame_count = 0
        self.score = 0
        self.lives = 3
        self.game_over = False
        self.timer = 60.0
        self.world_offset_x = 0
        
        self.last_safe_platform_idx = 0
        self.feedback_messages = []
        self.particles = []
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        
        # 1. Handle Input
        space_held = action[1] == 1
        jump_pressed = space_held and not self.prev_space_held
        
        if jump_pressed and self.on_ground:
            self.player_vel_y = self.JUMP_STRENGTH
            self.on_ground = False
            jump_reward, feedback = self._calculate_jump_timing_reward()
            reward += jump_reward
            if feedback:
                self.feedback_messages.append(feedback)
            self._create_particles(self.player_pos, 15, self.COLOR_PLAYER)
            # // Sound effect: Jump

        self.prev_space_held = space_held

        # 2. Update Physics & State
        self.player_vel_y += self.GRAVITY
        self.player_pos[1] += self.player_vel_y
        
        self.world_offset_x += self.SCROLL_SPEED
        self.frame_count += 1
        self.timer = max(0, self.timer - 1.0 / self.FPS)

        # 3. Collision Detection
        player_rect = pygame.Rect(self.player_pos[0], self.player_pos[1], *self.PLAYER_SIZE)
        landed_this_frame = False

        if self.player_vel_y > 0:
            for i, plat in enumerate(self.platforms):
                plat_screen_rect = plat.move(-self.world_offset_x, 0)
                if player_rect.colliderect(plat_screen_rect):
                    prev_player_bottom = player_rect.bottom - self.player_vel_y
                    if prev_player_bottom <= plat_screen_rect.top:
                        self.player_pos[1] = plat_screen_rect.top - player_rect.height
                        self.player_vel_y = 0
                        self.on_ground = True
                        landed_this_frame = True
                        if i > self.last_safe_platform_idx:
                            reward += 1.0  # Reward for landing on a new platform
                        self.last_safe_platform_idx = i
                        self._create_particles([player_rect.centerx, player_rect.bottom], 10, self.COLOR_PLATFORM_CYCLE[i % 3])
                        # // Sound effect: Land
                        break
        
        # 4. Check Termination Conditions & Penalties
        terminated = False
        if self.player_pos[1] > self.HEIGHT + 50:
            self.lives -= 1
            reward -= 5.0 # Penalty for falling
            if self.lives > 0:
                self._respawn_player()
            else:
                terminated = True
        
        if self.timer <= 0 and not terminated:
            terminated = True
        
        finish_line_screen_x = self.LEVEL_LENGTH - self.world_offset_x
        if finish_line_screen_x < self.player_pos[0] and not terminated:
            reward += 50.0
            terminated = True
        
        if terminated:
            self.game_over = True

        self.score += reward
        
        self._update_visual_effects()
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _calculate_jump_timing_reward(self):
        frames_into_beat = self.frame_count % self.FRAMES_PER_BEAT
        timing_error = min(frames_into_beat, self.FRAMES_PER_BEAT - frames_into_beat)
        
        player_center = [self.player_pos[0] + self.PLAYER_SIZE[0]/2, self.player_pos[1] - 20]

        if timing_error <= 1:
            feedback = {"text": "PERFECT!", "pos": player_center, "life": self.FPS, "color": self.COLOR_PERFECT}
            return 5.0, feedback
        elif timing_error <= 3:
            feedback = {"text": "Good", "pos": player_center, "life": self.FPS / 2, "color": self.COLOR_GOOD}
            return -2.0, feedback
        return 0, None

    def _respawn_player(self):
        safe_platform = self.platforms[self.last_safe_platform_idx]
        plat_screen_rect = safe_platform.move(-self.world_offset_x, 0)
        self.player_pos = [self.PLAYER_X_POS, plat_screen_rect.top - self.PLAYER_SIZE[1]]
        self.player_vel_y = 0
        self.on_ground = True
        self._create_particles(self.player_pos, 20, (255, 0, 0))
        # // Sound effect: Respawn

    def _update_visual_effects(self):
        # Update feedback messages
        self.feedback_messages = [m for m in self.feedback_messages if m['life'] > 0]
        for m in self.feedback_messages:
            m['life'] -= 1
            m['pos'][1] -= 0.5

        # Update particles
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['life'] -= 1
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]

    def _create_particles(self, pos, count, color):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            self.particles.append({
                "pos": list(pos),
                "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
                "life": random.randint(15, 30),
                "color": color
            })

    def _get_observation(self):
        self._draw_background()
        self._draw_finish_line()
        self._draw_platforms()
        self._draw_player()
        self._draw_particles()
        self._draw_feedback_messages()
        self._draw_beat_indicator()
        self._draw_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _draw_background(self):
        for y in range(self.HEIGHT):
            interp = y / self.HEIGHT
            color = (
                self.COLOR_BG_START[0] * (1 - interp) + self.COLOR_BG_END[0] * interp,
                self.COLOR_BG_START[1] * (1 - interp) + self.COLOR_BG_END[1] * interp,
                self.COLOR_BG_START[2] * (1 - interp) + self.COLOR_BG_END[2] * interp,
            )
            pygame.draw.line(self.screen, color, (0, y), (self.WIDTH, y))

    def _draw_finish_line(self):
        finish_x = self.LEVEL_LENGTH - self.world_offset_x
        if finish_x < self.WIDTH:
            for y in range(0, self.HEIGHT, 20):
                pygame.draw.rect(self.screen, self.COLOR_FINISH, (finish_x, y, 10, 10))
                pygame.draw.rect(self.screen, self.COLOR_FINISH, (finish_x + 10, y + 10, 10, 10))

    def _draw_platforms(self):
        beat_color_idx = int(self.frame_count / self.FRAMES_PER_BEAT)
        for i, plat in enumerate(self.platforms):
            plat_screen_rect = plat.move(-self.world_offset_x, 0)
            if plat_screen_rect.right < 0 or plat_screen_rect.left > self.WIDTH:
                continue

            color = self.COLOR_PLATFORM_CYCLE[(beat_color_idx + i) % 3]
            glow_color = (*color, 50)
            
            # Draw glow
            glow_rect = plat_screen_rect.inflate(8, 8)
            s = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
            pygame.draw.rect(s, glow_color, s.get_rect(), border_radius=8)
            self.screen.blit(s, glow_rect.topleft)

            # Draw platform
            pygame.draw.rect(self.screen, color, plat_screen_rect, border_radius=5)
            pygame.draw.rect(self.screen, (255,255,255), plat_screen_rect.inflate(-4,-4), width=1, border_radius=4)


    def _draw_player(self):
        player_rect = pygame.Rect(int(self.player_pos[0]), int(self.player_pos[1]), *self.PLAYER_SIZE)
        
        # Glow
        glow_size = player_rect.inflate(10, 10)
        s = pygame.Surface(glow_size.size, pygame.SRCALPHA)
        pygame.draw.ellipse(s, (*self.COLOR_PLAYER, 80), s.get_rect())
        self.screen.blit(s, glow_size.topleft)
        
        # Player body
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=3)
        
    def _draw_particles(self):
        for p in self.particles:
            alpha = int(255 * (p['life'] / 30))
            color = (*p['color'], alpha)
            pos = [int(p['pos'][0] - self.world_offset_x), int(p['pos'][1])]
            
            s = pygame.Surface((4, 4), pygame.SRCALPHA)
            pygame.draw.rect(s, color, s.get_rect())
            self.screen.blit(s, pos)

    def _draw_feedback_messages(self):
        for m in self.feedback_messages:
            alpha = int(255 * (m['life'] / self.FPS)) if m['life'] < self.FPS else 255
            text_surf = self.font_feedback.render(m['text'], True, m['color'])
            text_surf.set_alpha(alpha)
            pos = [int(m['pos'][0] - self.world_offset_x), int(m['pos'][1])]
            self.screen.blit(text_surf, text_surf.get_rect(center=pos))

    def _draw_beat_indicator(self):
        progress = (self.frame_count % self.FRAMES_PER_BEAT) / self.FRAMES_PER_BEAT
        pulse = abs(math.sin(progress * math.pi))
        
        radius = int(20 + 10 * pulse)
        alpha = int(50 + 200 * pulse)
        
        pos = (self.WIDTH // 2, self.HEIGHT - 30)
        
        # Outer glow
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, (*self.COLOR_PULSE, int(alpha/4)))
        # Inner circle
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(radius * 0.7), (*self.COLOR_PULSE, alpha))
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], int(radius * 0.7), (*self.COLOR_PULSE, alpha))

    def _draw_ui(self):
        # Score
        score_text = f"SCORE: {int(self.score)}"
        score_surf = self.font_ui.render(score_text, True, self.COLOR_UI)
        self.screen.blit(score_surf, (10, 10))
        
        # Timer
        timer_text = f"TIME: {self.timer:.1f}"
        timer_surf = self.font_ui.render(timer_text, True, self.COLOR_UI)
        self.screen.blit(timer_surf, (self.WIDTH - timer_surf.get_width() - 10, 10))
        
        # Lives
        lives_text = f"LIVES: {self.lives}"
        lives_surf = self.font_ui.render(lives_text, True, self.COLOR_UI)
        self.screen.blit(lives_surf, (10, 40))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.frame_count,
            "lives": self.lives,
            "timer": self.timer,
        }
        
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")