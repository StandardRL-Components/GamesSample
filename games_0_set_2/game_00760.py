
# Generated: 2025-08-27T14:41:19.485392
# Source Brief: brief_00760.md
# Brief Index: 760

        
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
        "Controls: ↑↓ to aim cannon, ←→ to adjust power. Press space to fire."
    )

    game_description = (
        "Side-view target practice. Use your limited ammo to hit moving targets. "
        "Hit 20 targets to win, or run out of ammo to lose."
    )

    auto_advance = True

    # --- Constants ---
    # Screen
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    
    # Colors
    COLOR_BG_SKY = (135, 206, 235)
    COLOR_BG_GROUND = (188, 224, 168)
    COLOR_CANNON = (70, 80, 90)
    COLOR_CANNON_BARREL = (100, 110, 120)
    COLOR_PROJECTILE = (34, 177, 76)
    COLOR_TARGET = (237, 28, 36)
    COLOR_EXPLOSION = [(255, 201, 14), (255, 127, 39), (255, 242, 0)]
    COLOR_UI_TEXT = (255, 255, 255)
    COLOR_UI_SHADOW = (50, 50, 50)
    COLOR_AIM_LINE = (255, 255, 255, 150)
    COLOR_POWER_BAR_BG = (50, 50, 50)
    COLOR_POWER_BAR_FG = (255, 215, 0)
    
    # Physics & Gameplay
    GRAVITY = 0.1
    MAX_STEPS = 1500 # Increased from 1000 to allow for travel time
    INITIAL_AMMO = 20
    WIN_CONDITION_HITS = 20
    FIRE_COOLDOWN = 15 # steps
    TARGET_RADIUS = 15
    NUM_TARGETS_ON_SCREEN = 4
    
    # Cannon
    CANNON_POS = (50, SCREEN_HEIGHT - 50)
    CANNON_ANGLE_MIN = -math.pi * 0.6
    CANNON_ANGLE_MAX = -math.pi * 0.05
    CANNON_ANGLE_STEP = 0.02
    CANNON_POWER_MIN = 20
    CANNON_POWER_MAX = 100
    CANNON_POWER_STEP = 2
    
    # Rewards
    REWARD_HIT = 0.5
    REWARD_MISS = -0.05
    REWARD_FAST_HIT = 2.0
    REWARD_WIN = 100.0
    REWARD_LOSE = -10.0
    
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
        
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_msg = pygame.font.SysFont("monospace", 40, bold=True)
        
        # Initialize state variables (done in reset)
        self.targets = []
        self.projectiles = []
        self.particles = []
        
        self.reset()
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0.0
        self.targets_hit = 0
        self.game_over = False
        self.win = False
        
        self.ammo = self.INITIAL_AMMO
        self.cannon_angle = (self.CANNON_ANGLE_MAX + self.CANNON_ANGLE_MIN) / 2
        self.cannon_power = (self.CANNON_POWER_MAX + self.CANNON_POWER_MIN) / 2
        
        self.projectiles.clear()
        self.particles.clear()
        self.targets.clear()
        
        self.target_base_speed = 1.0
        for _ in range(self.NUM_TARGETS_ON_SCREEN):
            self._spawn_target()
            
        self.last_fire_step = -self.FIRE_COOLDOWN
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.auto_advance:
            self.clock.tick(30)
            
        reward = 0.0
        
        if not self.game_over:
            self._handle_input(action)
            
            reward += self._update_projectiles()
            self._update_targets()
            self._update_particles()
        
        self.steps += 1
        terminated = self._check_termination()
        
        if terminated and not self.game_over:
            if self.win:
                reward += self.REWARD_WIN
            else: # Ran out of ammo or time
                reward += self.REWARD_LOSE
            self.game_over = True
            
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )
        
    def _handle_input(self, action):
        movement, space_held, _ = action
        
        # Aim
        if movement == 1: # Up
            self.cannon_angle -= self.CANNON_ANGLE_STEP
        elif movement == 2: # Down
            self.cannon_angle += self.CANNON_ANGLE_STEP
        self.cannon_angle = np.clip(self.cannon_angle, self.CANNON_ANGLE_MIN, self.CANNON_ANGLE_MAX)
        
        # Power
        if movement == 3: # Left
            self.cannon_power -= self.CANNON_POWER_STEP
        elif movement == 4: # Right
            self.cannon_power += self.CANNON_POWER_STEP
        self.cannon_power = np.clip(self.cannon_power, self.CANNON_POWER_MIN, self.CANNON_POWER_MAX)
        
        # Fire
        if space_held and self.ammo > 0 and (self.steps - self.last_fire_step) > self.FIRE_COOLDOWN:
            self.last_fire_step = self.steps
            self.ammo -= 1
            
            power_scalar = 0.2
            speed = self.cannon_power * power_scalar
            vel = np.array([
                math.cos(self.cannon_angle) * speed,
                math.sin(self.cannon_angle) * speed
            ], dtype=float)
            
            # sfx: cannon_fire.wav
            self.projectiles.append({
                "pos": np.array(self.CANNON_POS, dtype=float),
                "vel": vel,
                "trail": []
            })

    def _update_projectiles(self):
        step_reward = 0.0
        for proj in self.projectiles[:]:
            proj["vel"][1] += self.GRAVITY
            proj["pos"] += proj["vel"]
            
            proj["trail"].append(tuple(proj["pos"]))
            if len(proj["trail"]) > 15:
                proj["trail"].pop(0)

            # Check collision with targets
            hit_target = None
            for target in self.targets:
                dist = np.linalg.norm(proj["pos"] - target["pos"])
                if dist < target["radius"]:
                    hit_target = target
                    break
            
            if hit_target:
                # sfx: explosion.wav
                self._create_explosion(hit_target["pos"], self.COLOR_EXPLOSION)
                self.projectiles.remove(proj)
                self.targets.remove(hit_target)
                self.targets_hit += 1
                
                # Update difficulty
                self.target_base_speed = 1.0 + (self.targets_hit // 5) * 0.2
                
                fast_hit_threshold = 1.0 + (10 // 5) * 0.2 # Speed after 10 hits
                if abs(hit_target["speed"]) > fast_hit_threshold:
                    step_reward += self.REWARD_FAST_HIT
                else:
                    step_reward += self.REWARD_HIT
                
                self.score += step_reward
                self._spawn_target()
                continue

            # Check collision with ground/bounds
            if proj["pos"][1] > self.SCREEN_HEIGHT or not (0 < proj["pos"][0] < self.SCREEN_WIDTH):
                # sfx: fizzle.wav
                self.projectiles.remove(proj)
                step_reward += self.REWARD_MISS
                self.score += step_reward
        
        return step_reward

    def _update_targets(self):
        for target in self.targets:
            target["pos"][0] += target["speed"]
            if target["pos"][0] > self.SCREEN_WIDTH + self.TARGET_RADIUS:
                target["pos"][0] = -self.TARGET_RADIUS
            elif target["pos"][0] < -self.TARGET_RADIUS:
                target["pos"][0] = self.SCREEN_WIDTH + self.TARGET_RADIUS

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"] += p["vel"]
            p["life"] -= 1
            if p["life"] <= 0:
                self.particles.remove(p)

    def _check_termination(self):
        if self.targets_hit >= self.WIN_CONDITION_HITS:
            self.win = True
            return True
        if self.ammo <= 0 and not self.projectiles:
            return True
        if self.steps >= self.MAX_STEPS:
            return True
        return False

    def _get_observation(self):
        self._render_background()
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        ground_height = 50
        self.screen.fill(self.COLOR_BG_SKY)
        pygame.draw.rect(self.screen, self.COLOR_BG_GROUND, (0, self.SCREEN_HEIGHT - ground_height, self.SCREEN_WIDTH, ground_height))

    def _render_game(self):
        # Draw Cannon Base
        pygame.draw.circle(self.screen, self.COLOR_CANNON, (int(self.CANNON_POS[0]), int(self.CANNON_POS[1])), 20)
        
        # Draw Cannon Barrel
        barrel_length = 40
        end_x = self.CANNON_POS[0] + barrel_length * math.cos(self.cannon_angle)
        end_y = self.CANNON_POS[1] + barrel_length * math.sin(self.cannon_angle)
        pygame.draw.line(self.screen, self.COLOR_CANNON_BARREL, self.CANNON_POS, (int(end_x), int(end_y)), 10)

        # Draw Aiming Line if not game over
        if not self.game_over:
            self._render_aim_line()

        # Draw Targets
        for target in self.targets:
            pos = (int(target["pos"][0]), int(target["pos"][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], target["radius"], self.COLOR_TARGET)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], target["radius"], self.COLOR_TARGET)

        # Draw Projectiles
        for proj in self.projectiles:
            # Trail
            if len(proj["trail"]) > 1:
                for i in range(len(proj["trail"]) - 1):
                    alpha = int(200 * (i / len(proj["trail"])))
                    pygame.draw.line(self.screen, (*self.COLOR_PROJECTILE, alpha), proj["trail"][i], proj["trail"][i+1], 3)
            # Projectile glow
            pos = (int(proj["pos"][0]), int(proj["pos"][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 8, (*self.COLOR_PROJECTILE, 50))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 5, self.COLOR_PROJECTILE)

        # Draw Particles
        for p in self.particles:
            alpha = int(255 * (p["life"] / p["max_life"]))
            color = (*p["color"], alpha)
            size = int(p["size"] * (p["life"] / p["max_life"]))
            if size > 0:
                pygame.draw.circle(self.screen, color, (int(p["pos"][0]), int(p["pos"][1])), max(0, size))

    def _render_aim_line(self):
        pos = np.array(self.CANNON_POS, dtype=float)
        power_scalar = 0.2
        speed = self.cannon_power * power_scalar
        vel = np.array([math.cos(self.cannon_angle) * speed, math.sin(self.cannon_angle) * speed], dtype=float)
        
        for i in range(20):
            vel[1] += self.GRAVITY
            pos += vel
            if i % 2 == 0:
                pygame.draw.circle(self.screen, self.COLOR_AIM_LINE, (int(pos[0]), int(pos[1])), 2)

    def _render_ui(self):
        # Draw Power Bar
        power_ratio = (self.cannon_power - self.CANNON_POWER_MIN) / (self.CANNON_POWER_MAX - self.CANNON_POWER_MIN)
        bar_width, bar_height = 150, 20
        bar_x, bar_y = self.SCREEN_WIDTH // 2 - bar_width // 2, self.SCREEN_HEIGHT - 35
        pygame.draw.rect(self.screen, self.COLOR_POWER_BAR_BG, (bar_x, bar_y, bar_width, bar_height), border_radius=5)
        fill_width = max(0, power_ratio * (bar_width - 4))
        pygame.draw.rect(self.screen, self.COLOR_POWER_BAR_FG, (bar_x + 2, bar_y + 2, fill_width, bar_height - 4), border_radius=3)

        # Draw Score and Ammo
        score_text = f"SCORE: {self.targets_hit}/{self.WIN_CONDITION_HITS}"
        ammo_text = f"AMMO: {self.ammo}"
        self._draw_text(score_text, (20, 10), self.font_ui)
        self._draw_text(ammo_text, (self.SCREEN_WIDTH - 150, 10), self.font_ui)
        
        # Game Over / Win Message
        if self.game_over:
            msg = "YOU WIN!" if self.win else "GAME OVER"
            color = self.COLOR_PROJECTILE if self.win else self.COLOR_TARGET
            self._draw_text(msg, (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2), self.font_msg, color, center=True)

    def _draw_text(self, text, pos, font, color=COLOR_UI_TEXT, shadow_color=COLOR_UI_SHADOW, center=False):
        text_surf = font.render(text, True, color)
        shadow_surf = font.render(text, True, shadow_color)
        text_rect = text_surf.get_rect()
        if center:
            text_rect.center = pos
        else:
            text_rect.topleft = pos
        
        self.screen.blit(shadow_surf, (text_rect.x + 2, text_rect.y + 2))
        self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "ammo": self.ammo,
            "targets_hit": self.targets_hit,
        }

    def _spawn_target(self):
        side = self.np_random.choice([-1, 1])
        x = -self.TARGET_RADIUS if side == 1 else self.SCREEN_WIDTH + self.TARGET_RADIUS
        y = self.np_random.uniform(50, self.SCREEN_HEIGHT - 150)
        speed = self.target_base_speed * side * self.np_random.uniform(0.8, 1.2)
        
        self.targets.append({
            "pos": np.array([x, y], dtype=float),
            "radius": self.TARGET_RADIUS,
            "speed": speed
        })

    def _create_explosion(self, pos, colors):
        num_particles = 30
        for _ in range(num_particles):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 5)
            vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
            life = self.np_random.integers(15, 30)
            self.particles.append({
                "pos": np.array(pos, dtype=float),
                "vel": vel,
                "life": life,
                "max_life": life,
                "color": self.np_random.choice(colors),
                "size": self.np_random.integers(3, 7)
            })

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

if __name__ == '__main__':
    # To run and play the game manually
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    pygame.display.set_caption("Target Practice")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    running = True
    total_reward = 0
    
    while running:
        # Map keyboard keys to the MultiDiscrete action space
        keys = pygame.key.get_pressed()
        
        movement = 0 # No-op
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space = 1 if keys[pygame.K_SPACE] else 0
        shift = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation from the environment to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Total Reward: {total_reward:.2f}")
            pygame.time.wait(3000) # Pause for 3 seconds
            obs, info = env.reset()
            total_reward = 0
            
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

    pygame.quit()