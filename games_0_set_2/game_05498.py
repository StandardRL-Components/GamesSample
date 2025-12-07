
# Generated: 2025-08-28T05:12:57.532100
# Source Brief: brief_05498.md
# Brief Index: 5498

        
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

    user_guide = (
        "Controls: ←→ to walk. On a platform: ↑ for a small hop, hold Space for a high jump, or hold Shift for a long jump. Use ←→ to direct your jumps."
    )

    game_description = (
        "Leap between procedurally generated platforms to reach the top of the screen in this fast-paced arcade hopper."
    )

    auto_advance = True

    # --- Constants ---
    # Colors
    COLOR_BG_TOP = (40, 50, 90)
    COLOR_BG_BOTTOM = (10, 10, 30)
    COLOR_DANGER = (200, 0, 50, 100)
    COLOR_PLAYER = (255, 255, 255)
    COLOR_PLAYER_JUMP_GLOW = (180, 255, 255)
    COLOR_PLATFORM = (200, 200, 220)
    COLOR_PLATFORM_GOAL = (255, 223, 0)
    COLOR_TEXT = (240, 240, 240)
    COLOR_PARTICLE = (220, 220, 255)

    # Screen
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400

    # Player Physics
    GRAVITY = 0.5
    AIR_DRAG = 0.99
    AIR_CONTROL = 0.2
    WALK_SPEED = 2.5
    PLAYER_SIZE = (12, 18)

    # Jump Strengths
    JUMP_HOP = -8.5
    JUMP_HIGH = -12.5
    JUMP_LONG = -7.5
    JUMP_HIGH_X_MULT = 5
    JUMP_LONG_X_MULT = 7.5

    # Game Rules
    MAX_STEPS = 5000
    PLATFORM_COUNT = 15
    PLATFORM_HEIGHT = 12
    MAX_LIVES = 3

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
        self.font_ui = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_game_over = pygame.font.SysFont("Consolas", 48, bold=True)

        self.stage = 1
        self.lives = self.MAX_LIVES
        self.np_random = None

        self.reset()
        
        # self.validate_implementation() # Optional: Call for debugging

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.np_random is None:
            self.np_random = np.random.default_rng(seed)

        if self.lives <= 0:
            self.lives = self.MAX_LIVES
            self.stage = 1

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.victory = False

        self.player_pos = [self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT - 50]
        self.player_vel = [0, 0]
        self.on_platform = True
        self.jump_effect_timer = 0
        self.previous_player_y = self.player_pos[1]
        self.highest_platform_y = self.SCREEN_HEIGHT

        self.particles = []
        self._generate_platforms()

        # Ensure player starts on the first platform
        start_platform = self.platforms[0]['rect']
        self.player_pos[0] = start_platform.centerx
        self.player_pos[1] = start_platform.top - self.PLAYER_SIZE[1] / 2
        self.on_platform = start_platform

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.auto_advance:
            self.clock.tick(30)

        if self.game_over:
            # If game is over, no more actions, just wait for reset
            reward = 0
            terminated = True
            return self._get_observation(), reward, terminated, False, self._get_info()
            
        movement = action[0]
        space_held = action[1] == 1
        shift_held = action[2] == 1

        self.previous_player_y = self.player_pos[1]
        reward = 0

        self._handle_input(movement, space_held, shift_held)
        self._update_player_physics()
        self._update_platforms()
        self._update_particles()
        
        landed_platform = self._check_collisions()
        if landed_platform:
            # Reward for landing on a higher platform
            if landed_platform.top < self.highest_platform_y:
                reward += 1.0
                self.highest_platform_y = landed_platform.top

        # Continuous reward for upward movement
        height_gain = self.previous_player_y - self.player_pos[1]
        reward += height_gain * 0.01

        terminated = self._check_termination()
        if terminated:
            if self.victory:
                reward += 100.0
            elif self.player_pos[1] > self.SCREEN_HEIGHT + 20: # Fell off
                reward -= 5.0

        self.score += reward
        self.steps += 1

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info(),
        )

    def _generate_platforms(self):
        self.platforms = []
        plat_y_spacing = (self.SCREEN_HEIGHT - 100) / (self.PLATFORM_COUNT -1)
        
        # Start platform
        start_w = 120
        start_x = self.np_random.uniform(start_w / 2, self.SCREEN_WIDTH - start_w / 2)
        start_y = self.SCREEN_HEIGHT - 30
        self.platforms.append({
            "rect": pygame.Rect(start_x - start_w / 2, start_y, start_w, self.PLATFORM_HEIGHT),
            "base_y": start_y,
            "phase": self.np_random.uniform(0, 2 * math.pi),
            "is_goal": False
        })
        
        last_x = start_x
        for i in range(1, self.PLATFORM_COUNT):
            y = start_y - i * plat_y_spacing
            w = self.np_random.uniform(60, 90)
            max_dx = 150
            min_x = max(w / 2, last_x - max_dx)
            max_x = min(self.SCREEN_WIDTH - w / 2, last_x + max_dx)
            x = self.np_random.uniform(min_x, max_x)
            
            self.platforms.append({
                "rect": pygame.Rect(x - w/2, y, w, self.PLATFORM_HEIGHT),
                "base_y": y,
                "phase": self.np_random.uniform(0, 2 * math.pi),
                "is_goal": False
            })
            last_x = x

        # Goal platform
        goal_platform = self.platforms[-1]
        goal_platform['is_goal'] = True
        goal_platform['rect'].width = 150
        goal_platform['rect'].x = self.SCREEN_WIDTH / 2 - 75

    def _handle_input(self, movement, space_held, shift_held):
        # --- Jumping ---
        if self.on_platform:
            jump_initiated = False
            jump_power_y = 0
            jump_power_x_mult = 0
            
            # Priority: Space > Shift > Up Arrow
            if space_held:
                jump_initiated = True
                jump_power_y = self.JUMP_HIGH
                jump_power_x_mult = self.JUMP_HIGH_X_MULT
                # sfx: big_jump
            elif shift_held:
                jump_initiated = True
                jump_power_y = self.JUMP_LONG
                jump_power_x_mult = self.JUMP_LONG_X_MULT
                # sfx: long_jump
            elif movement == 1: # Up
                jump_initiated = True
                jump_power_y = self.JUMP_HOP
                # sfx: small_hop

            if jump_initiated:
                self.player_vel[1] = jump_power_y
                if movement == 3: # Left
                    self.player_vel[0] = -jump_power_x_mult
                elif movement == 4: # Right
                    self.player_vel[0] = jump_power_x_mult
                self.on_platform = None
                self.jump_effect_timer = 8 # frames

        # --- Movement ---
        if self.on_platform:
            # Walk on platform
            if movement == 3: # Left
                self.player_pos[0] -= self.WALK_SPEED
            elif movement == 4: # Right
                self.player_pos[0] += self.WALK_SPEED
        else:
            # Air control
            if movement == 3: # Left
                self.player_vel[0] -= self.AIR_CONTROL
            elif movement == 4: # Right
                self.player_vel[0] += self.AIR_CONTROL

    def _update_player_physics(self):
        if self.on_platform:
            # Stick to platform
            self.player_vel = [0, 0]
            self.player_pos[1] = self.on_platform.top - self.PLAYER_SIZE[1] / 2
            # Check if walked off
            if not self.on_platform.collidepoint(self.player_pos[0], self.on_platform.top):
                self.on_platform = None
        else:
            # Apply gravity
            self.player_vel[1] += self.GRAVITY
            # Apply air drag
            self.player_vel[0] *= self.AIR_DRAG
            # Update position
            self.player_pos[0] += self.player_vel[0]
            self.player_pos[1] += self.player_vel[1]

        # Screen bounds (horizontal)
        self.player_pos[0] = np.clip(self.player_pos[0], self.PLAYER_SIZE[0]/2, self.SCREEN_WIDTH - self.PLAYER_SIZE[0]/2)
        if self.player_pos[0] <= self.PLAYER_SIZE[0]/2 or self.player_pos[0] >= self.SCREEN_WIDTH - self.PLAYER_SIZE[0]/2:
            self.player_vel[0] = 0

    def _update_platforms(self):
        oscillation_speed = 0.5 + (self.stage - 1) * 0.05
        amplitude = 3
        for plat in self.platforms:
            plat['rect'].y = plat['base_y'] + math.sin(plat['phase'] + self.steps * 0.1 * oscillation_speed) * amplitude

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1

    def _check_collisions(self):
        if self.player_vel[1] < 0: # Moving up
            return None

        player_rect = pygame.Rect(
            self.player_pos[0] - self.PLAYER_SIZE[0]/2,
            self.player_pos[1] - self.PLAYER_SIZE[1]/2,
            self.PLAYER_SIZE[0], self.PLAYER_SIZE[1]
        )
        
        for plat in self.platforms:
            p_rect = plat['rect']
            # Check if player's bottom is intersecting the top surface of the platform
            if (p_rect.colliderect(player_rect) and
                player_rect.bottom < p_rect.bottom and
                self.previous_player_y + self.PLAYER_SIZE[1]/2 <= p_rect.top):
                
                self.on_platform = p_rect
                self.player_pos[1] = p_rect.top - self.PLAYER_SIZE[1] / 2
                self.player_vel = [0, 0]
                self._create_landing_particles(p_rect.midtop)
                # sfx: land
                
                if plat['is_goal']:
                    self.victory = True
                
                return p_rect
        return None

    def _check_termination(self):
        if self.victory:
            self.stage += 1
            self.game_over = True
            return True

        if self.player_pos[1] > self.SCREEN_HEIGHT + 20:
            self.lives -= 1
            self.game_over = True
            # sfx: fall_and_lose_life
            return True

        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True
            
        return False

    def _create_landing_particles(self, pos):
        for _ in range(10):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3)
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed * 0.5],
                'life': self.np_random.integers(10, 20),
                'size': self.np_random.uniform(2, 4)
            })

    def _get_observation(self):
        self._render_background()
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "stage": self.stage,
        }

    def _render_background(self):
        for y in range(self.SCREEN_HEIGHT):
            interp = y / self.SCREEN_HEIGHT
            color = (
                self.COLOR_BG_BOTTOM[0] + interp * (self.COLOR_BG_TOP[0] - self.COLOR_BG_BOTTOM[0]),
                self.COLOR_BG_BOTTOM[1] + interp * (self.COLOR_BG_TOP[1] - self.COLOR_BG_BOTTOM[1]),
                self.COLOR_BG_BOTTOM[2] + interp * (self.COLOR_BG_TOP[2] - self.COLOR_BG_BOTTOM[2]),
            )
            pygame.draw.line(self.screen, color, (0, y), (self.SCREEN_WIDTH, y))

        # Danger zone
        danger_surface = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT / 5), pygame.SRCALPHA)
        danger_surface.fill(self.COLOR_DANGER)
        self.screen.blit(danger_surface, (0, self.SCREEN_HEIGHT * 4 / 5))

    def _render_game(self):
        # Render platforms
        for plat in self.platforms:
            color = self.COLOR_PLATFORM_GOAL if plat['is_goal'] else self.COLOR_PLATFORM
            pygame.draw.rect(self.screen, color, plat['rect'], border_radius=3)
            # Add a subtle 3D effect
            pygame.draw.rect(self.screen, tuple(max(0, c-30) for c in color), plat['rect'].move(0, 4), border_radius=3)


        # Render particles
        for p in self.particles:
            alpha = p['life'] / 20
            color = (*self.COLOR_PARTICLE, int(255 * alpha))
            pygame.gfxdraw.filled_circle(
                self.screen, int(p['pos'][0]), int(p['pos'][1]), int(p['size'] * alpha), color
            )

        # Render player
        px, py = int(self.player_pos[0]), int(self.player_pos[1])
        pw, ph = self.PLAYER_SIZE
        
        points = [
            (px, py - ph / 2),
            (px - pw / 2, py + ph / 2),
            (px + pw / 2, py + ph / 2),
        ]
        
        if self.jump_effect_timer > 0:
            self.jump_effect_timer -= 1
            glow_size = 1 + (self.jump_effect_timer / 8) * 1.5
            glow_points = [
                (px, py - ph/2 * glow_size),
                (px - pw/2 * glow_size, py + ph/2 * glow_size),
                (px + pw/2 * glow_size, py + ph/2 * glow_size),
            ]
            alpha = int(150 * (self.jump_effect_timer / 8))
            pygame.gfxdraw.aapolygon(self.screen, glow_points, (*self.COLOR_PLAYER_JUMP_GLOW, alpha))

        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)

    def _render_ui(self):
        # Render lives
        heart_size = 10
        for i in range(self.lives):
            x, y = 20 + i * (heart_size * 2), 20
            points = [
                (x, y + heart_size * 0.25),
                (x - heart_size/2, y - heart_size * 0.25),
                (x - heart_size/4, y - heart_size * 0.6),
                (x, y - heart_size * 0.25),
                (x + heart_size/4, y - heart_size * 0.6),
                (x + heart_size/2, y - heart_size * 0.25),
            ]
            pygame.gfxdraw.aapolygon(self.screen, points, (255, 80, 80))
            pygame.gfxdraw.filled_polygon(self.screen, points, (255, 80, 80))

        # Render score/height
        height_text = f"HEIGHT: {int(self.SCREEN_HEIGHT - self.player_pos[1])}"
        text_surf = self.font_ui.render(height_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surf, (self.SCREEN_WIDTH - text_surf.get_width() - 15, 10))

        # Render Stage
        stage_text = f"STAGE: {self.stage}"
        stage_surf = self.font_ui.render(stage_text, True, self.COLOR_TEXT)
        self.screen.blit(stage_surf, ((self.SCREEN_WIDTH - stage_surf.get_width()) / 2, 10))

        # Render Game Over / Victory
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))

            if self.victory:
                msg = f"STAGE {self.stage-1} CLEAR!"
            elif self.lives <= 0:
                msg = "GAME OVER"
            else: # Fell or timed out
                msg = "TRY AGAIN"

            text_surf = self.font_game_over.render(msg, True, self.COLOR_TEXT)
            self.screen.blit(text_surf, ((self.SCREEN_WIDTH - text_surf.get_width()) / 2, 
                                         (self.SCREEN_HEIGHT - text_surf.get_height()) / 2))

    def close(self):
        pygame.quit()

    def validate_implementation(self):
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