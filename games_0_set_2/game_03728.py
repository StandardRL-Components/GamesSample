
# Generated: 2025-08-28T00:15:03.145400
# Source Brief: brief_03728.md
# Brief Index: 3728

        
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
        "Controls: Press space to jump. Timing is everything."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A minimalist platformer. Ascend procedurally generated levels by "
        "jumping between platforms. Land on red platforms for bonus points before they disappear."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen_width = 640
        self.screen_height = 400
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_msg = pygame.font.SysFont("monospace", 36, bold=True)
        
        # --- Game Constants ---
        self.GRAVITY = 0.35
        self.JUMP_STRENGTH = -8.0
        self.PLAYER_SIZE = 20
        self.MAX_STEPS = 10000

        # --- Colors ---
        self.COLOR_BG_TOP = (173, 216, 230)
        self.COLOR_BG_BOTTOM = (25, 25, 112)
        self.COLOR_PLAYER = (0, 191, 255)
        self.COLOR_PLAYER_OUTLINE = (255, 255, 255)
        self.COLOR_PLATFORM = (110, 110, 110)
        self.COLOR_PLATFORM_OUTLINE = (50, 50, 50)
        self.COLOR_RISKY_PLATFORM = (255, 69, 0)
        self.COLOR_GOAL_PLATFORM = (255, 215, 0)
        self.COLOR_TEXT = (255, 255, 255)

        # --- Reward Structure ---
        self.REWARD_SURVIVAL = 0.01 # Changed from 0.1 to keep rewards smaller
        self.REWARD_NORMAL_LAND = 5.0
        self.REWARD_RISKY_LAND = 10.0
        self.REWARD_PENALTY = -1.0
        self.REWARD_LEVEL_COMPLETE = 50.0
        
        # Initialize state variables
        self.player_pos = pygame.Vector2(0, 0)
        self.player_vel = pygame.Vector2(0, 0)
        self.on_ground = False
        self.platforms = []
        self.particles = []
        self.camera_y = 0
        self.level = 1
        self.timer = 0.0
        self.rng = np.random.default_rng()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.level_complete_frames = 0
        
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.rng = np.random.default_rng(seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.level_complete_frames = 0
        self.level = 1
        self.timer = 0.0
        
        self.platforms = []
        self.particles = []
        
        self._generate_level()

        self.player_pos = pygame.Vector2(self.screen_width / 2, self.platforms[0].pos.y - self.PLAYER_SIZE * 2)
        self.player_vel = pygame.Vector2(0, 0)
        self.on_ground = True
        
        self.camera_y = self.player_pos.y - self.screen_height * 0.7
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0.0, True, False, self._get_info()

        reward = 0.0
        self.steps += 1
        self.timer += 1 / 30.0
        
        # --- Handle Level Completion Transition ---
        if self.level_complete_frames > 0:
            self.level_complete_frames -= 1
            if self.level_complete_frames == 0:
                self.level += 1
                self.timer = 0.0
                self._generate_level()
                self.player_pos = pygame.Vector2(self.screen_width / 2, self.platforms[0].pos.y - self.PLAYER_SIZE * 2)
                self.player_vel = pygame.Vector2(0, 0)
                self.on_ground = True
            # Return early during transition
            return self._get_observation(), reward, False, False, self._get_info()

        # --- Unpack Action & Handle Input ---
        space_held = action[1] == 1
        
        if space_held and self.on_ground:
            self.player_vel.y = self.JUMP_STRENGTH
            self.on_ground = False
            # sfx: jump

        # --- Physics & Game Logic ---
        # Player physics
        self.player_vel.y += self.GRAVITY
        self.player_pos += self.player_vel
        self.on_ground = False 

        # Platform movement
        for p in self.platforms:
            p.update()

        # --- Collision Detection ---
        landed_on_safe = False
        landed_on_risky = False

        if self.player_vel.y > 0: # Only check for landing when falling
            player_rect = pygame.Rect(self.player_pos.x - self.PLAYER_SIZE / 2, self.player_pos.y - self.PLAYER_SIZE, self.PLAYER_SIZE, self.PLAYER_SIZE)
            prev_player_bottom = player_rect.bottom - self.player_vel.y

            for p in self.platforms:
                if player_rect.colliderect(p.rect) and prev_player_bottom <= p.rect.top:
                    self.player_pos.y = p.rect.top
                    self.player_vel.y = 0
                    self.on_ground = True
                    self._create_particles(pygame.Vector2(self.player_pos.x, self.player_pos.y), p.color)
                    # sfx: land

                    if not p.landed_on:
                        p.landed_on = True
                        if p.type == 'goal':
                            reward += self.REWARD_LEVEL_COMPLETE
                            self.level_complete_frames = 45 # 1.5 second pause
                        elif p.type == 'risky':
                            reward += self.REWARD_RISKY_LAND
                            p.is_disappearing = True
                            landed_on_risky = True
                        else: # 'normal'
                            reward += self.REWARD_NORMAL_LAND
                            landed_on_safe = True
                    break

        # --- Reward Penalty Logic ---
        if landed_on_safe:
            risky_was_available = any(
                p.type == 'risky' and not p.is_disappearing and 0 < p.pos.y - self.camera_y < self.screen_height
                for p in self.platforms
            )
            if risky_was_available:
                reward += self.REWARD_PENALTY

        # --- Update World State ---
        self.platforms = [p for p in self.platforms if not p.to_remove]
        self._update_particles()
        
        # Smooth camera follow
        target_camera_y = self.player_pos.y - self.screen_height * 0.6
        self.camera_y += (target_camera_y - self.camera_y) * 0.08
        
        # --- Final Rewards & Termination ---
        reward += self.REWARD_SURVIVAL
        self.score += reward
        
        terminated = False
        if self.player_pos.y - self.camera_y > self.screen_height + 50:
            terminated = True
            self.game_over = True
            # sfx: fall_death
        
        if self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _generate_level(self):
        self.platforms.clear()
        num_obstacles = 9 + self.level
        platform_base_speed = 0.5 + (self.level - 1) * 0.5
        risky_chance = min(0.5, 0.05 + 0.05 * self.level)
        
        # Base platform
        base_y = self.player_pos.y + 100 if self.level > 1 else 350
        p = Platform(
            pos=pygame.Vector2(self.screen_width / 2 - 75, base_y),
            size=pygame.Vector2(150, 20),
            p_type='normal',
            color=self.COLOR_PLATFORM
        )
        self.platforms.append(p)
        
        current_y = base_y
        last_x = self.screen_width / 2

        # Procedurally generated platforms
        for i in range(num_obstacles):
            dy = self.rng.integers(70, 130)
            dx_max = 180
            dx = self.rng.uniform(-dx_max, dx_max)
            
            current_y -= dy
            current_x = np.clip(last_x + dx, 100, self.screen_width - 100)
            last_x = current_x

            is_risky = self.rng.random() < risky_chance
            p_type = 'risky' if is_risky else 'normal'
            width = self.rng.integers(50, 70) if is_risky else self.rng.integers(80, 130)
            color = self.COLOR_RISKY_PLATFORM if is_risky else self.COLOR_PLATFORM
            
            speed = 0
            if self.rng.random() < 0.2 + 0.05 * self.level:
                speed = self.rng.uniform(0.5, platform_base_speed) * (1 if self.rng.random() < 0.5 else -1)

            p = Platform(
                pos=pygame.Vector2(current_x - width / 2, current_y),
                size=pygame.Vector2(width, 20),
                p_type=p_type,
                color=color,
                vel=pygame.Vector2(speed, 0)
            )
            self.platforms.append(p)

        # Goal platform
        goal_y = current_y - 150
        p = Platform(
            pos=pygame.Vector2(self.screen_width / 2 - 50, goal_y),
            size=pygame.Vector2(100, 25),
            p_type='goal',
            color=self.COLOR_GOAL_PLATFORM
        )
        self.platforms.append(p)

    def _create_particles(self, pos, color):
        for _ in range(15):
            vel = pygame.Vector2(self.rng.uniform(-2, 2), self.rng.uniform(-3, 0))
            self.particles.append(Particle(pos, vel, color, self.rng.integers(15, 30)))

    def _update_particles(self):
        for p in self.particles:
            p.update()
        self.particles = [p for p in self.particles if p.lifespan > 0]
        
    def _get_observation(self):
        self._draw_gradient_background()
        
        # Render game elements
        for p in self.particles:
            p.draw(self.screen, self.camera_y)
            
        for p in self.platforms:
            p.draw(self.screen, self.camera_y)
        
        # Render player
        player_screen_pos = pygame.Vector2(self.player_pos.x, self.player_pos.y - self.camera_y)
        player_rect = pygame.Rect(
            int(player_screen_pos.x - self.PLAYER_SIZE / 2),
            int(player_screen_pos.y - self.PLAYER_SIZE),
            self.PLAYER_SIZE, self.PLAYER_SIZE
        )
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=3)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER_OUTLINE, player_rect, width=2, border_radius=3)
        
        # Render UI overlay
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _draw_gradient_background(self):
        for y in range(self.screen_height):
            ratio = y / self.screen_height
            color = (
                int(self.COLOR_BG_TOP[0] * (1 - ratio) + self.COLOR_BG_BOTTOM[0] * ratio),
                int(self.COLOR_BG_TOP[1] * (1 - ratio) + self.COLOR_BG_BOTTOM[1] * ratio),
                int(self.COLOR_BG_TOP[2] * (1 - ratio) + self.COLOR_BG_BOTTOM[2] * ratio)
            )
            pygame.draw.line(self.screen, color, (0, y), (self.screen_width, y))

    def _render_ui(self):
        level_text = self.font_ui.render(f"LEVEL: {self.level}", True, self.COLOR_TEXT)
        self.screen.blit(level_text, (10, 10))

        timer_text = self.font_ui.render(f"TIME: {self.timer:.2f}", True, self.COLOR_TEXT)
        self.screen.blit(timer_text, (self.screen_width - timer_text.get_width() - 10, 10))

        score_text = self.font_ui.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 35))

        if self.level_complete_frames > 0:
            msg = self.font_msg.render("LEVEL COMPLETE!", True, self.COLOR_GOAL_PLATFORM)
            msg_rect = msg.get_rect(center=(self.screen_width / 2, self.screen_height / 2))
            self.screen.blit(msg, msg_rect)
        elif self.game_over:
            msg = self.font_msg.render("GAME OVER", True, self.COLOR_RISKY_PLATFORM)
            msg_rect = msg.get_rect(center=(self.screen_width / 2, self.screen_height / 2))
            self.screen.blit(msg, msg_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "level": self.level,
            "timer": self.timer,
        }

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

class Particle:
    def __init__(self, pos, vel, color, lifespan):
        self.pos = pygame.Vector2(pos)
        self.vel = pygame.Vector2(vel)
        self.color = color
        self.lifespan = lifespan
        self.max_lifespan = lifespan
        self.size = self.lifespan / 4

    def update(self):
        self.pos += self.vel
        self.vel.y += 0.1 # particle gravity
        self.lifespan -= 1
        self.size = max(0, self.lifespan / 4)

    def draw(self, surface, camera_y):
        alpha = int(255 * (self.lifespan / self.max_lifespan))
        temp_surf = pygame.Surface((self.size*2, self.size*2), pygame.SRCALPHA)
        
        pygame.draw.circle(
            temp_surf,
            (*self.color, alpha),
            (self.size, self.size),
            self.size
        )
        
        screen_pos_x = int(self.pos.x - self.size)
        screen_pos_y = int(self.pos.y - camera_y - self.size)
        surface.blit(temp_surf, (screen_pos_x, screen_pos_y))

class Platform:
    def __init__(self, pos, size, p_type, color, vel=pygame.Vector2(0, 0)):
        self.pos = pos
        self.size = size
        self.type = p_type
        self.color = color
        self.outline_color = tuple(max(0, c - 60) for c in color)
        self.vel = vel
        self.rect = pygame.Rect(pos.x, pos.y, size.x, size.y)
        self.landed_on = False
        self.is_disappearing = False
        self.disappear_timer = 30 # 1 second
        self.to_remove = False

    def update(self):
        if self.vel.x != 0:
            self.pos.x += self.vel.x
            if self.pos.x < 0 or self.pos.x + self.size.x > 640:
                self.vel.x *= -1
            self.rect.x = self.pos.x
        
        if self.is_disappearing:
            self.disappear_timer -= 1
            if self.disappear_timer <= 0:
                self.to_remove = True

    def draw(self, surface, camera_y):
        screen_y = self.pos.y - camera_y
        self.rect.y = screen_y
        
        alpha = 255
        if self.is_disappearing:
            alpha = max(0, int(255 * (self.disappear_timer / 30)))

        if alpha < 255:
            # Use a temporary surface for transparency
            temp_surf = pygame.Surface(self.size, pygame.SRCALPHA)
            temp_surf.set_alpha(alpha)
            pygame.draw.rect(temp_surf, self.color, (0, 0, self.size.x, self.size.y), border_radius=4)
            pygame.draw.rect(temp_surf, self.outline_color, (0, 0, self.size.x, self.size.y), width=2, border_radius=4)
            surface.blit(temp_surf, (self.pos.x, screen_y))
        else:
            # Opaque drawing is faster
            rect_to_draw = pygame.Rect(int(self.pos.x), int(screen_y), int(self.size.x), int(self.size.y))
            pygame.draw.rect(surface, self.color, rect_to_draw, border_radius=4)
            pygame.draw.rect(surface, self.outline_color, rect_to_draw, width=2, border_radius=4)

if __name__ == '__main__':
    # To run and play the game manually
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Set up Pygame window for display
    screen = pygame.display.set_mode((640, 400))
    pygame.display.set_caption("Minimalist Platformer")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        action = [0, 0, 0] # [movement, space, shift]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE]:
            action[1] = 1

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Display the observation from the environment
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Steps: {info['steps']}")
            # Wait for a moment before auto-resetting
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0
            
        clock.tick(30) # Match the environment's step rate

    pygame.quit()