import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T10:18:20.402419
# Source Brief: brief_00389.md
# Brief Index: 389
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    An arcade-style Gymnasium environment where the player launches projectiles
    to hit moving targets. The game features different projectile types,
    power-ups, and a dynamic difficulty system. The goal is to achieve a
    high score by skillfully aiming and managing power.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Aim your cannon and launch projectiles to destroy moving targets. "
        "Collect power-ups for special abilities and aim for a high score."
    )
    user_guide = (
        "Controls: Use ↑/↓ to adjust power and ←/→ to aim the cannon. Press space to fire."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    WIN_SCORE = 50
    MAX_STEPS = 5000
    TARGET_SPEED_INCREASE_THRESHOLD = 5 # Speed increases every 5 hits
    
    # Colors
    COLOR_BG = (20, 25, 35)
    COLOR_GRID = (35, 40, 50)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_LAUNCHER_BASE = (100, 110, 130)
    COLOR_LAUNCHER_BARREL = (180, 190, 210)
    
    COLOR_TARGET = (50, 255, 100)
    COLOR_POWERUP = (200, 80, 255)
    
    PROJECTILE_COLORS = {
        "standard": (255, 80, 80),
        "explosive": (255, 200, 80),
        "piercing": (80, 150, 255)
    }
    
    # Physics & Gameplay
    GRAVITY = 0.15
    LAUNCHER_POS = pygame.Vector2(60, SCREEN_HEIGHT - 50)
    POWER_CHANGE_RATE = 0.2
    ANGLE_CHANGE_RATE = 1.0  # degrees
    FIRE_COOLDOWN_STEPS = 10
    
    # Rewards
    REWARD_HIT_TARGET = 0.1
    REWARD_MISS = -0.01
    REWARD_COLLECT_POWERUP = 1.0
    REWARD_WIN = 100.0

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Gymnasium Spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame Setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 16, bold=True)
        
        # State variables are initialized in reset()
        self.reset()

        # self.validate_implementation() # Optional self-check
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Game State
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        # Player/Launcher State
        self.launcher_angle = 45.0  # degrees
        self.launcher_power = 5.0   # 1-10 range
        self.fire_cooldown = 0
        self.prev_space_held = False
        
        # Projectile & Power-up State
        self.projectile_type = "standard"
        self.powerup_duration = 0
        
        # Entity Lists
        self.projectiles = []
        self.targets = []
        self.powerups = []
        self.particles = []
        self.explosions = []

        # Difficulty Scaling
        self.target_speed = 1.0
        self.hits_since_speed_increase = 0

        self._spawn_initial_targets()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        # --- 1. Handle Input ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # Adjust launcher power (Up/Down)
        if movement == 1: self.launcher_power += self.POWER_CHANGE_RATE
        if movement == 2: self.launcher_power -= self.POWER_CHANGE_RATE
        self.launcher_power = np.clip(self.launcher_power, 1.0, 10.0)

        # Adjust launcher angle (Left/Right)
        if movement == 3: self.launcher_angle += self.ANGLE_CHANGE_RATE
        if movement == 4: self.launcher_angle -= self.ANGLE_CHANGE_RATE
        self.launcher_angle = np.clip(self.launcher_angle, 0.0, 90.0)
        
        # Launch projectile (Space press)
        if space_held and not self.prev_space_held and self.fire_cooldown == 0:
            self._launch_projectile()
            self.fire_cooldown = self.FIRE_COOLDOWN_STEPS
            # Placeholder for sound effect: pygame.mixer.Sound("launch.wav").play()
        self.prev_space_held = space_held
        if self.fire_cooldown > 0: self.fire_cooldown -= 1

        # --- 2. Update Game Logic & Calculate Rewards ---
        step_reward = 0.0
        
        self._update_entities()
        
        # Handle collisions and accumulate rewards
        collision_rewards = self._handle_collisions()
        step_reward += collision_rewards

        # Spawn new entities
        self._maybe_spawn_powerup()
        if len(self.targets) < 3: self._spawn_target()

        # --- 3. Update Global State ---
        self.steps += 1
        if self.powerup_duration > 0:
            self.powerup_duration -= 1
            if self.powerup_duration == 0:
                self.projectile_type = "standard"

        # --- 4. Check Termination & Final Reward ---
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        if terminated and not truncated: # Win condition
            step_reward += self.REWARD_WIN
        
        return (
            self._get_observation(),
            step_reward,
            terminated,
            truncated,
            self._get_info()
        )

    # --- Update Methods ---
    def _update_entities(self):
        """Update positions and states of all dynamic entities."""
        # Update projectiles
        for p in self.projectiles:
            p['vel'].y += self.GRAVITY
            p['pos'] += p['vel']
        
        # Update targets
        for t in self.targets:
            t['pos'].x += t['vel_x']
            if t['pos'].x > self.SCREEN_WIDTH + t['radius']:
                t['pos'].x = -t['radius']
                t['pos'].y = random.randint(50, self.SCREEN_HEIGHT - 150)
            elif t['pos'].x < -t['radius']:
                t['pos'].x = self.SCREEN_WIDTH + t['radius']
                t['pos'].y = random.randint(50, self.SCREEN_HEIGHT - 150)

        # Update explosions
        for e in self.explosions:
            e['radius'] += e['expansion_rate']
            e['life'] -= 1
        
        # Update particles
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1
            p['radius'] = max(0, p['radius'] - 0.1)

        # Cleanup dead entities
        self.projectiles = [p for p in self.projectiles if 0 < p['pos'].x < self.SCREEN_WIDTH and p['pos'].y < self.SCREEN_HEIGHT]
        self.particles = [p for p in self.particles if p['life'] > 0]
        self.explosions = [e for e in self.explosions if e['life'] > 0]

    def _handle_collisions(self):
        """Check and process all collisions, returning rewards."""
        reward = 0
        projectiles_to_remove = set()
        targets_to_remove = set()
        powerups_to_remove = set()

        # Projectile -> Target/Powerup
        for i, proj in enumerate(self.projectiles):
            if i in projectiles_to_remove: continue
            
            # Projectile -> Target
            for j, target in enumerate(self.targets):
                if j in targets_to_remove: continue
                if proj['pos'].distance_to(target['pos']) < proj['radius'] + target['radius']:
                    targets_to_remove.add(j)
                    reward += self._on_target_hit(target)
                    
                    if proj['type'] == 'standard':
                        projectiles_to_remove.add(i)
                    elif proj['type'] == 'explosive':
                        self._create_explosion(proj['pos'])
                        projectiles_to_remove.add(i)
                    elif proj['type'] == 'piercing':
                        proj['pierce_count'] -= 1
                        if proj['pierce_count'] <= 0:
                            projectiles_to_remove.add(i)
                    break # Projectile can only hit one target per frame
            
            # Projectile -> Powerup
            for k, powerup in enumerate(self.powerups):
                if k in powerups_to_remove: continue
                if proj['pos'].distance_to(powerup['pos']) < proj['radius'] + powerup['radius']:
                    powerups_to_remove.add(k)
                    projectiles_to_remove.add(i)
                    reward += self._on_powerup_collect(powerup)
                    break

        # Explosion -> Target
        for explosion in self.explosions:
            for j, target in enumerate(self.targets):
                if j in targets_to_remove: continue
                if explosion['pos'].distance_to(target['pos']) < explosion['radius'] + target['radius']:
                    targets_to_remove.add(j)
                    reward += self._on_target_hit(target)

        # Reward for missed projectiles
        num_proj_before = len(self.projectiles)
        
        # Perform removals
        self.projectiles = [p for i, p in enumerate(self.projectiles) if i not in projectiles_to_remove]
        self.targets = [t for j, t in enumerate(self.targets) if j not in targets_to_remove]
        self.powerups = [pu for k, pu in enumerate(self.powerups) if k not in powerups_to_remove]
        
        num_proj_after = len(self.projectiles)
        reward += (num_proj_after - num_proj_before) * self.REWARD_MISS

        return reward

    def _on_target_hit(self, target):
        """Actions to take when a target is hit."""
        self.score += 1
        self.hits_since_speed_increase += 1
        self._spawn_particle_explosion(target['pos'], self.COLOR_TARGET, 20, 3)
        # Placeholder for sound effect: pygame.mixer.Sound("hit.wav").play()
        
        if self.hits_since_speed_increase >= self.TARGET_SPEED_INCREASE_THRESHOLD:
            self.hits_since_speed_increase = 0
            self.target_speed = min(10.0, self.target_speed + 0.1)
            for t in self.targets:
                t['vel_x'] = self.target_speed * np.sign(t['vel_x'])
        
        return self.REWARD_HIT_TARGET

    def _on_powerup_collect(self, powerup):
        """Actions to take when a powerup is collected."""
        self.projectile_type = powerup['type']
        self.powerup_duration = 300 # 10 seconds at 30fps
        self._spawn_particle_explosion(powerup['pos'], self.COLOR_POWERUP, 30, 4)
        # Placeholder for sound effect: pygame.mixer.Sound("powerup.wav").play()
        return self.REWARD_COLLECT_POWERUP

    # --- Spawning Methods ---
    def _spawn_initial_targets(self):
        for _ in range(3):
            self._spawn_target()

    def _spawn_target(self):
        y_pos = random.randint(50, self.SCREEN_HEIGHT - 150)
        x_pos = random.choice([0, self.SCREEN_WIDTH])
        vel_x = self.target_speed if x_pos == 0 else -self.target_speed
        self.targets.append({
            'pos': pygame.Vector2(x_pos, y_pos),
            'vel_x': vel_x,
            'radius': 15
        })

    def _maybe_spawn_powerup(self):
        if not self.powerups and random.random() < 0.005: # 0.5% chance per frame
            pos = pygame.Vector2(random.randint(200, self.SCREEN_WIDTH - 100), random.randint(50, self.SCREEN_HEIGHT - 200))
            ptype = random.choice(["explosive", "piercing"])
            self.powerups.append({
                'pos': pos,
                'radius': 12,
                'type': ptype,
            })

    def _spawn_particle_explosion(self, pos, color, count, max_speed):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, max_speed)
            vel = pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed)
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'life': random.randint(20, 40),
                'color': color,
                'radius': random.uniform(2, 5)
            })

    def _launch_projectile(self):
        angle_rad = math.radians(self.launcher_angle)
        vel = pygame.Vector2(
            math.cos(angle_rad) * self.launcher_power,
            -math.sin(angle_rad) * self.launcher_power
        )
        
        barrel_end_pos = self.LAUNCHER_POS + pygame.Vector2(35, 0).rotate(-self.launcher_angle)
        
        proj = {
            'pos': barrel_end_pos,
            'vel': vel,
            'radius': 6,
            'type': self.projectile_type
        }
        if self.projectile_type == 'piercing':
            proj['pierce_count'] = 3
        
        self.projectiles.append(proj)

    def _create_explosion(self, pos):
        self.explosions.append({
            'pos': pos.copy(),
            'radius': 10,
            'max_radius': 60,
            'life': 10,
            'expansion_rate': 6
        })
        self._spawn_particle_explosion(pos, self.PROJECTILE_COLORS['explosive'], 40, 5)
        # Placeholder for sound effect: pygame.mixer.Sound("explosion.wav").play()

    # --- Rendering Methods ---
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_entities()
        self._render_launcher()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for i in range(0, self.SCREEN_WIDTH, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (i, 0), (i, self.SCREEN_HEIGHT))
        for i in range(0, self.SCREEN_HEIGHT, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, i), (self.SCREEN_WIDTH, i))

    def _render_launcher(self):
        # Base
        pygame.draw.circle(self.screen, self.COLOR_LAUNCHER_BASE, (int(self.LAUNCHER_POS.x), int(self.LAUNCHER_POS.y)), 20)
        
        # Barrel
        barrel_vec = pygame.Vector2(35, 0).rotate(-self.launcher_angle)
        p1 = self.LAUNCHER_POS + barrel_vec.normalize() * 5
        p2 = self.LAUNCHER_POS + barrel_vec
        pygame.draw.line(self.screen, self.COLOR_LAUNCHER_BARREL, (int(p1.x), int(p1.y)), (int(p2.x), int(p2.y)), 12)
        
        # Power bar
        bar_height = 100
        bar_width = 15
        bar_x = self.LAUNCHER_POS.x - 40
        bar_y = self.LAUNCHER_POS.y - bar_height
        fill_height = (self.launcher_power / 10.0) * bar_height
        pygame.draw.rect(self.screen, self.COLOR_GRID, (bar_x, bar_y, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.PROJECTILE_COLORS['standard'], (bar_x, bar_y + bar_height - fill_height, bar_width, fill_height))

    def _render_entities(self):
        # Explosions
        for e in self.explosions:
            alpha = int(255 * (e['life'] / 10))
            self._draw_glowing_circle(self.screen, self.PROJECTILE_COLORS['explosive'], e['pos'], e['radius'], glow_factor=1.2, alpha=alpha)

        # Targets
        for t in self.targets:
            self._draw_glowing_circle(self.screen, self.COLOR_TARGET, t['pos'], t['radius'])

        # Powerups
        for pu in self.powerups:
            self._draw_glowing_circle(self.screen, self.COLOR_POWERUP, pu['pos'], pu['radius'])
            # Draw icon inside
            icon_color = self.PROJECTILE_COLORS[pu['type']]
            if pu['type'] == 'explosive':
                pygame.draw.circle(self.screen, icon_color, (int(pu['pos'].x), int(pu['pos'].y)), int(pu['radius'] * 0.6))
            elif pu['type'] == 'piercing':
                p1 = pu['pos'] + pygame.Vector2(-pu['radius']*0.5, 0)
                p2 = pu['pos'] + pygame.Vector2(pu['radius']*0.5, 0)
                pygame.draw.line(self.screen, icon_color, (int(p1.x), int(p1.y)), (int(p2.x), int(p2.y)), 3)

        # Projectiles
        for p in self.projectiles:
            color = self.PROJECTILE_COLORS[p['type']]
            self._draw_glowing_circle(self.screen, color, p['pos'], p['radius'])

        # Particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / 40.0))
            color = p['color'] + (alpha,)
            s = pygame.Surface((p['radius']*2, p['radius']*2), pygame.SRCALPHA)
            pygame.draw.circle(s, color, (p['radius'], p['radius']), p['radius'])
            self.screen.blit(s, (int(p['pos'].x - p['radius']), int(p['pos'].y - p['radius'])))

    def _render_ui(self):
        # Score
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        # Powerup Status
        if self.powerup_duration > 0:
            color = self.PROJECTILE_COLORS[self.projectile_type]
            text = self.font_small.render(f"{self.projectile_type.upper()} ACTIVE", True, color)
            self.screen.blit(text, (10, 40))
            
            # Duration bar
            bar_width = text.get_width()
            bar_height = 5
            fill_width = (self.powerup_duration / 300.0) * bar_width
            pygame.draw.rect(self.screen, self.COLOR_GRID, (10, 60, bar_width, bar_height))
            pygame.draw.rect(self.screen, color, (10, 60, fill_width, bar_height))

    def _draw_glowing_circle(self, surface, color, center, radius, glow_factor=1.5, alpha=100):
        """Draws a circle with a soft, glowing aura."""
        center_int = (int(center.x), int(center.y))
        
        # Glow
        glow_radius = int(radius * glow_factor)
        s = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        glow_color = color + (alpha,)
        pygame.draw.circle(s, glow_color, (glow_radius, glow_radius), glow_radius)
        surface.blit(s, (center_int[0] - glow_radius, center_int[1] - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)
        
        # Core circle (anti-aliased)
        pygame.gfxdraw.aacircle(surface, center_int[0], center_int[1], int(radius), color)
        pygame.gfxdraw.filled_circle(surface, center_int[0], center_int[1], int(radius), color)

    # --- Gymnasium Interface Methods ---
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "launcher_power": self.launcher_power,
            "launcher_angle": self.launcher_angle,
            "active_projectile": self.projectile_type,
            "target_speed": self.target_speed,
        }
    
    def _check_termination(self):
        return self.score >= self.WIN_SCORE

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
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # --- Manual Play Example ---
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Override Pygame display for manual playing
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Manual Control - Projectile Power")
    clock = pygame.time.Clock()
    
    terminated = False
    truncated = False
    total_reward = 0

    while not (terminated or truncated):
        # Action mapping for human player
        keys = pygame.key.get_pressed()
        movement = 0 # none
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = keys[pygame.K_SPACE]
        shift_held = keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]
        
        action = [movement, 1 if space_held else 0, 1 if shift_held else 0]

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        clock.tick(30) # Run at 30 FPS

    print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
    env.close()