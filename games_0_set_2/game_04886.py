
# Generated: 2025-08-28T03:19:36.931377
# Source Brief: brief_04886.md
# Brief Index: 4886

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
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

    user_guide = (
        "Controls: Use ← and → to select jump direction. Hold 'Space' for a high jump or 'Shift' for a short hop. "
        "Press ↑ for a vertical jump."
    )

    game_description = (
        "Guide your space hopper ever upwards, leaping between precarious platforms to reach the summit. "
        "Reach higher levels to reset the timer. Don't fall!"
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.render_mode = render_mode
        self.screen_width = 640
        self.screen_height = 400

        self.observation_space = Box(
            low=0, high=255, shape=(self.screen_height, self.screen_width, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 32)
        self.font_title = pygame.font.Font(None, 50)

        # --- Colors ---
        self.COLOR_BG_TOP = (10, 0, 30)
        self.COLOR_BG_BOTTOM = (40, 0, 70)
        self.COLOR_PLAYER = (255, 60, 60)
        self.COLOR_PLAYER_GLOW = (255, 150, 150)
        self.COLOR_PLATFORM_EDGE = (50, 255, 150)
        self.COLOR_PLATFORM_SURFACE = (40, 200, 120)
        self.COLOR_GOAL_PLATFORM = (255, 223, 0)
        self.COLOR_PARTICLE = (255, 255, 100)
        self.COLOR_UI_TEXT = (255, 255, 255)
        self.STAR_COLORS = [(200, 200, 255), (255, 255, 255), (255, 220, 220)]

        # --- Game Physics & Constants ---
        self.GRAVITY = 0.4
        self.FRICTION = 0.98
        self.PLAYER_SIZE = 16
        self.JUMP_POWER_HIGH = -11
        self.JUMP_POWER_NORMAL = -8.5
        self.JUMP_POWER_LOW = -6
        self.HORIZONTAL_SPEED = 5.5
        self.MAX_STEPS = 2700  # 90 seconds at 30 FPS

        # --- State Variables (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.hopper_pos = None
        self.hopper_vel = None
        self.is_grounded = False
        self.platforms = []
        self.original_platform_dims = []
        self.particles = []
        self.stars = []
        self.max_height_idx = 0
        self.level_timer = 0
        self.level_timer_max = 600 # 20 seconds * 30 FPS
        self.levels_cleared = 0
        self.level_heights = []

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.is_grounded = True
        self.particles.clear()
        
        self._generate_platforms()
        
        self.hopper_pos = [self.platforms[0].centerx, self.platforms[0].top - self.PLAYER_SIZE / 2]
        self.hopper_vel = [0, 0]

        self.max_height_idx = 0
        self.level_timer = self.level_timer_max
        self.levels_cleared = 0
        
        if not self.stars:
            self._generate_stars()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0.01  # Survival reward

        self._handle_input(movement, space_held, shift_held)
        self._update_physics()
        
        landed_on_platform, platform_idx = self._check_collisions()
        
        if landed_on_platform:
            # Sound: land.wav
            self.is_grounded = True
            self.hopper_vel = [0, 0]
            self.hopper_pos[1] = self.platforms[platform_idx].top - self.PLAYER_SIZE / 2
            self._create_particles(self.platforms[platform_idx].midtop, 15)
            
            reward += 1.0

            if platform_idx > self.max_height_idx:
                reward += (platform_idx - self.max_height_idx) * 2.0
                self.score += (platform_idx - self.max_height_idx) * 10
                self.max_height_idx = platform_idx
                
                # Check for level progression
                if self.levels_cleared < len(self.level_heights) and self.hopper_pos[1] < self.level_heights[self.levels_cleared]:
                    self.levels_cleared += 1
                    self.level_timer = self.level_timer_max
                    # Sound: level_up.wav
                    self._create_particles(self.hopper_pos, 30, color=(100, 255, 100))
                    reward += 10.0
                    self.score += 100

        self._update_difficulty()
        self._update_particles()
        
        self.steps += 1
        self.level_timer -= 1
        
        terminated, term_reward = self._check_termination(platform_idx)
        reward += term_reward
        self.game_over = terminated
        
        if self.game_over and term_reward > 0:
            # Sound: win.wav
            self._create_particles(self.hopper_pos, 100, color=(255, 223, 0))
        elif self.game_over and term_reward < 0:
            # Sound: fall.wav
            pass

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, movement, space_held, shift_held):
        if self.is_grounded and movement in [1, 3, 4]:
            self.is_grounded = False
            # Sound: jump.wav
            
            jump_power = self.JUMP_POWER_NORMAL
            if space_held:
                jump_power = self.JUMP_POWER_HIGH
            elif shift_held:
                jump_power = self.JUMP_POWER_LOW
            
            self.hopper_vel[1] = jump_power
            
            if movement == 3: # Left
                self.hopper_vel[0] = -self.HORIZONTAL_SPEED
            elif movement == 4: # Right
                self.hopper_vel[0] = self.HORIZONTAL_SPEED

    def _update_physics(self):
        if not self.is_grounded:
            self.hopper_vel[1] += self.GRAVITY
        
        self.hopper_vel[0] *= self.FRICTION
        
        self.hopper_pos[0] += self.hopper_vel[0]
        self.hopper_pos[1] += self.hopper_vel[1]

        # Screen wrapping for horizontal movement
        if self.hopper_pos[0] < 0:
            self.hopper_pos[0] = self.screen_width
        elif self.hopper_pos[0] > self.screen_width:
            self.hopper_pos[0] = 0

    def _check_collisions(self):
        hopper_rect = pygame.Rect(
            self.hopper_pos[0] - self.PLAYER_SIZE / 2,
            self.hopper_pos[1] - self.PLAYER_SIZE / 2,
            self.PLAYER_SIZE, self.PLAYER_SIZE
        )
        
        if self.hopper_vel[1] > 0: # Only check for landing if moving down
            for i, plat in enumerate(self.platforms):
                if hopper_rect.colliderect(plat) and hopper_rect.bottom < plat.bottom:
                    return True, i
        return False, -1

    def _check_termination(self, platform_idx):
        if self.hopper_pos[1] > self.screen_height + self.PLAYER_SIZE:
            return True, -5.0 # Fell off screen
        if self.level_timer <= 0:
            return True, -1.0 # Timed out
        if self.steps >= self.MAX_STEPS:
            return True, 0.0 # Max steps reached
        if platform_idx == len(self.platforms) - 1:
            self.score += 1000
            return True, 100.0 # Reached goal
        return False, 0.0

    def _generate_platforms(self):
        self.platforms.clear()
        self.original_platform_dims.clear()
        
        # Base platform
        base_plat = pygame.Rect(50, 360, self.screen_width - 100, 20)
        self.platforms.append(base_plat)
        
        num_platforms = 12
        y_pos = base_plat.y
        
        for i in range(1, num_platforms):
            width = max(60, 150 - i * 8)
            x_pos = self.np_random.integers(0, self.screen_width - width)
            y_pos -= self.np_random.integers(70, 95)
            
            plat = pygame.Rect(x_pos, y_pos, width, 15)
            self.platforms.append(plat)
        
        self.original_platform_dims = [(p.width, p.height) for p in self.platforms]
        
        # Define level thresholds based on platform heights
        self.level_heights = [
            self.platforms[num_platforms // 3].y,
            self.platforms[2 * num_platforms // 3].y,
        ]

    def _update_difficulty(self):
        scale = max(0.5, 1.0 - 0.05 * (self.steps // 500))
        if scale < 1.0:
            for i, plat in enumerate(self.platforms[1:], 1): # Don't shrink base
                original_width = self.original_platform_dims[i][0]
                new_width = original_width * scale
                plat.x += (plat.width - new_width) / 2
                plat.width = new_width

    def _generate_stars(self):
        self.stars.clear()
        for _ in range(100):
            self.stars.append(
                (
                    self.np_random.integers(0, self.screen_width),
                    self.np_random.integers(0, self.screen_height),
                    self.np_random.integers(1, 4), # size
                    self.np_random.choice(self.STAR_COLORS)
                )
            )

    def _create_particles(self, pos, count, color=None):
        if color is None:
            color = self.COLOR_PARTICLE
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed - 2]
            lifespan = self.np_random.integers(15, 30)
            self.particles.append({'pos': list(pos), 'vel': vel, 'life': lifespan, 'color': color})

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # particle gravity
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _get_observation(self):
        self._render_background()
        self._render_game_elements()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        # Gradient background
        for y in range(self.screen_height):
            ratio = y / self.screen_height
            color = (
                int(self.COLOR_BG_TOP[0] * (1 - ratio) + self.COLOR_BG_BOTTOM[0] * ratio),
                int(self.COLOR_BG_TOP[1] * (1 - ratio) + self.COLOR_BG_BOTTOM[1] * ratio),
                int(self.COLOR_BG_TOP[2] * (1 - ratio) + self.COLOR_BG_BOTTOM[2] * ratio),
            )
            pygame.draw.line(self.screen, color, (0, y), (self.screen_width, y))
        
        # Stars
        for x, y, size, color in self.stars:
            pygame.draw.rect(self.screen, color, (x, y, size, size))

    def _render_game_elements(self):
        # Platforms
        for i, plat in enumerate(self.platforms):
            is_goal = i == len(self.platforms) - 1
            edge_color = self.COLOR_GOAL_PLATFORM if is_goal else self.COLOR_PLATFORM_EDGE
            surf_color = self.COLOR_GOAL_PLATFORM if is_goal else self.COLOR_PLATFORM_SURFACE
            
            # Draw filled surface then outline for crisp edge
            pygame.draw.rect(self.screen, surf_color, plat, border_radius=3)
            pygame.draw.rect(self.screen, edge_color, plat, width=2, border_radius=3)
        
        # Particles
        for p in self.particles:
            size = max(1, int(p['life'] / 6))
            pygame.draw.rect(self.screen, p['color'], (int(p['pos'][0]), int(p['pos'][1]), size, size))

        # Player
        if not self.game_over or self.hopper_pos[1] < self.screen_height:
            pos_x, pos_y = int(self.hopper_pos[0]), int(self.hopper_pos[1])
            size = self.PLAYER_SIZE
            
            # Glow effect
            glow_radius = int(size * 1.5)
            glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, (*self.COLOR_PLAYER_GLOW, 80), (glow_radius, glow_radius), glow_radius)
            self.screen.blit(glow_surf, (pos_x - glow_radius, pos_y - glow_radius))

            # Player body
            player_rect = pygame.Rect(pos_x - size/2, pos_y - size/2, size, size)
            pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=3)
            pygame.draw.rect(self.screen, (255, 255, 255), player_rect.inflate(-6, -6), border_radius=2)


    def _render_ui(self):
        score_text = self.font_ui.render(f"Score: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (15, 10))
        
        timer_val = self.level_timer / 30
        timer_color = (255, 100, 100) if timer_val < 5 else self.COLOR_UI_TEXT
        timer_text = self.font_ui.render(f"Time: {timer_val:.1f}", True, timer_color)
        self.screen.blit(timer_text, (self.screen_width - timer_text.get_width() - 15, 10))

        if self.game_over:
            outcome_text_str = "GOAL!" if self.max_height_idx == len(self.platforms) - 1 else "GAME OVER"
            outcome_text = self.font_title.render(outcome_text_str, True, self.COLOR_UI_TEXT)
            text_rect = outcome_text.get_rect(center=(self.screen_width / 2, self.screen_height / 2))
            self.screen.blit(outcome_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "max_height_idx": self.max_height_idx,
            "is_grounded": self.is_grounded,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.screen_height, self.screen_width, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (self.screen_height, self.screen_width, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.screen_height, self.screen_width, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    env = GameEnv(render_mode="rgb_array")
    
    # --- Pygame setup for human play ---
    human_screen = pygame.display.set_mode((env.screen_width, env.screen_height))
    pygame.display.set_caption(env.game_description)
    clock = pygame.time.Clock()
    
    obs, info = env.reset()
    terminated = False
    
    print(env.user_guide)

    while not terminated:
        # --- Human Controls ---
        keys = pygame.key.get_pressed()
        movement = 0 # none
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        action = [movement, space_held, shift_held]

        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Rendering for Human ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        human_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event Handling & Clock ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        clock.tick(30) # Match the auto_advance rate

        if terminated:
            print(f"Game Over! Final Score: {info['score']}")
            # Wait a bit before closing
            pygame.time.wait(3000)

    env.close()