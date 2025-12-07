
# Generated: 2025-08-27T21:45:19.879094
# Source Brief: brief_02896.md
# Brief Index: 2896

        
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
        "Controls: Arrow keys to move and aim. Hold Shift to reload. Press Space to fire your weapon."
    )

    game_description = (
        "Survive waves of procedurally generated zombies in a top-down arena by strategically managing ammo and aiming shots."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.ARENA_PADDING = 20

        # Player
        self.PLAYER_SIZE = 10
        self.PLAYER_SPEED = 4
        self.PLAYER_HEALTH_MAX = 100
        self.PLAYER_AMMO_MAX = 20
        self.PLAYER_HIT_COOLDOWN = 60  # 2s invulnerability

        # Zombie
        self.ZOMBIE_SIZE = 8
        self.ZOMBIE_BASE_SPEED = 0.6
        self.ZOMBIE_HEALTH_MAX = 3
        self.ZOMBIE_DAMAGE = 10
        self.ZOMBIE_HIT_COOLDOWN = 10 # short stun on hit

        # Gameplay
        self.BULLET_SIZE = 3
        self.BULLET_SPEED = 12
        self.MAX_WAVES = 5
        self.MAX_STEPS = 4500 # 150 seconds at 30 FPS
        self.RELOAD_TIME = 60 # 2 seconds
        self.SHOOT_COOLDOWN = 8 # frames

        # --- Colors ---
        self.COLOR_BG = (20, 25, 30)
        self.COLOR_ARENA = (40, 50, 60)
        self.COLOR_PLAYER = (50, 150, 255)
        self.COLOR_PLAYER_GLOW = (100, 180, 255)
        self.COLOR_ZOMBIE = (0, 120, 50)
        self.COLOR_ZOMBIE_DAMAGED = (150, 100, 50)
        self.COLOR_BULLET = (255, 255, 0)
        self.COLOR_UI_TEXT = (220, 220, 220)
        self.COLOR_UI_HEALTH = (220, 30, 30)
        self.COLOR_UI_AMMO = (255, 200, 0)
        self.COLOR_UI_BAR_BG = (80, 80, 80)
        self.COLOR_PARTICLE_BLOOD = (180, 0, 0)
        self.COLOR_PARTICLE_SPARK = (255, 255, 150)

        # --- Gymnasium Spaces ---
        self.observation_space = Box(low=0, high=255, shape=(400, 640, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("monospace", 36, bold=True)
        self.font_medium = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 16)

        # --- State Variables ---
        self.player_pos = None
        self.player_health = None
        self.player_ammo = None
        self.player_aim_angle = None
        self.player_hit_cooldown_timer = None
        self.is_reloading = None
        self.reload_timer = None
        self.shoot_cooldown_timer = None
        self.screen_shake_timer = None

        self.zombies = None
        self.bullets = None
        self.particles = None

        self.wave = None
        self.zombies_to_spawn_this_wave = None
        self.score = None
        self.steps = None
        self.game_over = None
        self.game_won = None
        
        self.np_random = None

        self.reset()
        self.validate_implementation()


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.player_pos = np.array([self.WIDTH / 2, self.HEIGHT / 2], dtype=float)
        self.player_health = self.PLAYER_HEALTH_MAX
        self.player_ammo = self.PLAYER_AMMO_MAX
        self.player_aim_angle = 0.0  # Radians, 0 is right
        self.player_hit_cooldown_timer = 0
        self.is_reloading = False
        self.reload_timer = 0
        self.shoot_cooldown_timer = 0
        self.screen_shake_timer = 0

        self.zombies = []
        self.bullets = []
        self.particles = []

        self.wave = 0
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.game_won = False

        self._start_new_wave()

        return self._get_observation(), self._get_info()

    def _start_new_wave(self):
        self.wave += 1
        if self.wave > self.MAX_WAVES:
            self.game_won = True
            return

        self.zombies_to_spawn_this_wave = 25 + self.wave * 5
        for _ in range(self.zombies_to_spawn_this_wave):
            side = self.np_random.integers(4)
            if side == 0:  # Top
                pos = [self.np_random.uniform(0, self.WIDTH), -self.ZOMBIE_SIZE]
            elif side == 1:  # Bottom
                pos = [self.np_random.uniform(0, self.WIDTH), self.HEIGHT + self.ZOMBIE_SIZE]
            elif side == 2:  # Left
                pos = [-self.ZOMBIE_SIZE, self.np_random.uniform(0, self.HEIGHT)]
            else:  # Right
                pos = [self.WIDTH + self.ZOMBIE_SIZE, self.np_random.uniform(0, self.HEIGHT)]
            
            self.zombies.append({
                "pos": np.array(pos, dtype=float),
                "health": self.ZOMBIE_HEALTH_MAX,
                "speed": self.ZOMBIE_BASE_SPEED + (self.wave - 1) * 0.05,
                "hit_cooldown": 0
            })

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0.0

        # --- 1. Handle Player Actions ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # Reload
        if shift_held and not self.is_reloading and self.player_ammo < self.PLAYER_AMMO_MAX:
            self.is_reloading = True
            self.reload_timer = self.RELOAD_TIME
            # sfx: reload_start

        # Movement & Aiming (disabled during reload)
        if not self.is_reloading:
            move_vector = np.array([0.0, 0.0])
            if movement == 1:  # Up
                move_vector[1] = -1
                self.player_aim_angle = -math.pi / 2
            elif movement == 2:  # Down
                move_vector[1] = 1
                self.player_aim_angle = math.pi / 2
            elif movement == 3:  # Left
                move_vector[0] = -1
                self.player_aim_angle = math.pi
            elif movement == 4:  # Right
                move_vector[0] = 1
                self.player_aim_angle = 0
            
            if np.linalg.norm(move_vector) > 0:
                self.player_pos += move_vector * self.PLAYER_SPEED

        # Shooting (disabled during reload)
        if space_held and not self.is_reloading and self.player_ammo > 0 and self.shoot_cooldown_timer <= 0:
            self.player_ammo -= 1
            self.shoot_cooldown_timer = self.SHOOT_COOLDOWN
            # sfx: shoot_laser
            bullet_pos = self.player_pos + np.array([math.cos(self.player_aim_angle), math.sin(self.player_aim_angle)]) * (self.PLAYER_SIZE + 1)
            bullet_vel = np.array([math.cos(self.player_aim_angle), math.sin(self.player_aim_angle)]) * self.BULLET_SPEED
            self.bullets.append({"pos": bullet_pos, "vel": bullet_vel})
            self._create_particles(bullet_pos, self.COLOR_PARTICLE_SPARK, count=5, speed_mult=1.5, life=8)
        elif space_held and self.player_ammo == 0 and not self.is_reloading and self.shoot_cooldown_timer <= 0:
            self.shoot_cooldown_timer = self.SHOOT_COOLDOWN # Cooldown for empty click
            # sfx: empty_click

        # --- 2. Update Game State ---
        # Timers
        self.shoot_cooldown_timer = max(0, self.shoot_cooldown_timer - 1)
        self.player_hit_cooldown_timer = max(0, self.player_hit_cooldown_timer - 1)
        self.screen_shake_timer = max(0, self.screen_shake_timer - 1)
        if self.is_reloading:
            self.reload_timer -= 1
            if self.reload_timer <= 0:
                self.is_reloading = False
                self.player_ammo = self.PLAYER_AMMO_MAX
                # sfx: reload_complete
        
        # Zombies
        for z in self.zombies:
            z['hit_cooldown'] = max(0, z['hit_cooldown'] - 1)
            direction = self.player_pos - z['pos']
            dist = np.linalg.norm(direction)
            if dist > 1:
                z['pos'] += (direction / dist) * z['speed']

        # Bullets & Particles
        for b in self.bullets: b['pos'] += b['vel']
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1
            p['radius'] = max(0, p['radius'] - 0.1)
        self.particles = [p for p in self.particles if p['life'] > 0 and p['radius'] > 0]

        # --- 3. Handle Collisions & Rewards ---
        # Bullet-Zombie Collisions
        bullets_to_remove = set()
        zombies_to_remove = set()
        for i, b in enumerate(self.bullets):
            for j, z in enumerate(self.zombies):
                if i in bullets_to_remove or j in zombies_to_remove: continue
                if np.linalg.norm(b['pos'] - z['pos']) < self.ZOMBIE_SIZE + self.BULLET_SIZE and z['hit_cooldown'] <= 0:
                    bullets_to_remove.add(i)
                    z['health'] -= 1
                    z['hit_cooldown'] = self.ZOMBIE_HIT_COOLDOWN
                    reward += 0.1  # Hit reward
                    self._create_particles(z['pos'], self.COLOR_PARTICLE_BLOOD, count=15, speed_mult=2, life=20)
                    # sfx: zombie_hit
                    if z['health'] <= 0:
                        zombies_to_remove.add(j)
                        self.score += 10
                        reward += 1.0  # Kill reward
                        # sfx: zombie_die
                    break # Bullet hits one zombie
        
        # Missed shots
        for i, b in enumerate(self.bullets):
            if i not in bullets_to_remove:
                if not (0 < b['pos'][0] < self.WIDTH and 0 < b['pos'][1] < self.HEIGHT):
                    bullets_to_remove.add(i)
                    reward -= 0.01 # Miss penalty
        
        self.bullets = [b for i, b in enumerate(self.bullets) if i not in bullets_to_remove]
        self.zombies = [z for i, z in enumerate(self.zombies) if i not in zombies_to_remove]

        # Player-Zombie Collisions
        if self.player_hit_cooldown_timer <= 0:
            for z in self.zombies:
                if np.linalg.norm(self.player_pos - z['pos']) < self.PLAYER_SIZE + self.ZOMBIE_SIZE:
                    self.player_health -= self.ZOMBIE_DAMAGE
                    self.player_hit_cooldown_timer = self.PLAYER_HIT_COOLDOWN
                    self.screen_shake_timer = 15
                    reward -= 1.0  # Damage penalty
                    # sfx: player_hurt
                    break

        # --- 4. Game Progression ---
        if len(self.zombies) == 0 and self.zombies_to_spawn_this_wave == 0:
            if self.wave < self.MAX_WAVES:
                reward += 100.0  # Wave clear reward
                self._start_new_wave()
            else:
                self.game_won = True

        # --- 5. Finalize State & Termination ---
        self.player_pos[0] = np.clip(self.player_pos[0], self.ARENA_PADDING, self.WIDTH - self.ARENA_PADDING)
        self.player_pos[1] = np.clip(self.player_pos[1], self.ARENA_PADDING, self.HEIGHT - self.ARENA_PADDING)
        self.player_health = max(0, self.player_health)

        terminated = False
        if self.player_health <= 0:
            terminated = True
            self.game_over = True
            reward = -100.0
        elif self.game_won:
            terminated = True
            self.game_over = True
            reward += 100.0 # Win bonus
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _get_observation(self):
        # Determine render offset for screen shake
        offset_x, offset_y = 0, 0
        if self.screen_shake_timer > 0:
            offset_x = self.np_random.integers(-5, 6)
            offset_y = self.np_random.integers(-5, 6)

        # Clear screen
        self.screen.fill(self.COLOR_BG)

        # Render game elements with offset
        self._render_game(offset_x, offset_y)

        # Render UI elements without offset
        self._render_ui()

        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self, ox, oy):
        # Arena
        arena_rect = pygame.Rect(self.ARENA_PADDING + ox, self.ARENA_PADDING + oy,
                                 self.WIDTH - 2 * self.ARENA_PADDING, self.HEIGHT - 2 * self.ARENA_PADDING)
        pygame.draw.rect(self.screen, self.COLOR_ARENA, arena_rect, border_radius=5)

        # Particles
        for p in self.particles:
            pos = (int(p['pos'][0] + ox), int(p['pos'][1] + oy))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(p['radius']), p['color'])

        # Zombies
        for z in self.zombies:
            pos = (int(z['pos'][0] + ox), int(z['pos'][1] + oy))
            color = self.COLOR_ZOMBIE_DAMAGED if z['hit_cooldown'] > 0 else self.COLOR_ZOMBIE
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.ZOMBIE_SIZE, color)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.ZOMBIE_SIZE, color)

        # Player
        player_pos_int = (int(self.player_pos[0] + ox), int(self.player_pos[1] + oy))
        
        # Player flashing when hit
        is_invincible = self.player_hit_cooldown_timer > 0
        if not (is_invincible and self.steps % 4 < 2):
            # Glow effect
            glow_surf = pygame.Surface((self.PLAYER_SIZE * 4, self.PLAYER_SIZE * 4), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, (*self.COLOR_PLAYER_GLOW, 60), (self.PLAYER_SIZE * 2, self.PLAYER_SIZE * 2), self.PLAYER_SIZE * 2)
            self.screen.blit(glow_surf, (player_pos_int[0] - self.PLAYER_SIZE * 2, player_pos_int[1] - self.PLAYER_SIZE * 2), special_flags=pygame.BLEND_RGBA_ADD)
            
            # Player body
            pygame.gfxdraw.filled_circle(self.screen, player_pos_int[0], player_pos_int[1], self.PLAYER_SIZE, self.COLOR_PLAYER)
            pygame.gfxdraw.aacircle(self.screen, player_pos_int[0], player_pos_int[1], self.PLAYER_SIZE, self.COLOR_PLAYER)

            # Aiming indicator
            end_pos_x = player_pos_int[0] + math.cos(self.player_aim_angle) * (self.PLAYER_SIZE + 5)
            end_pos_y = player_pos_int[1] + math.sin(self.player_aim_angle) * (self.PLAYER_SIZE + 5)
            pygame.draw.line(self.screen, self.COLOR_PLAYER_GLOW, player_pos_int, (int(end_pos_x), int(end_pos_y)), 3)

        # Bullets
        for b in self.bullets:
            pos = (int(b['pos'][0] + ox), int(b['pos'][1] + oy))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.BULLET_SIZE, self.COLOR_BULLET)

    def _render_ui(self):
        # Health Bar
        health_ratio = self.player_health / self.PLAYER_HEALTH_MAX
        bar_width = 200
        bar_height = 20
        pygame.draw.rect(self.screen, self.COLOR_UI_BAR_BG, (10, 10, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_UI_HEALTH, (10, 10, int(bar_width * health_ratio), bar_height))
        health_text = self.font_small.render("HP", True, self.COLOR_UI_TEXT)
        self.screen.blit(health_text, (10 + bar_width / 2 - health_text.get_width() / 2, 12))

        # Ammo Bar
        ammo_ratio = self.player_ammo / self.PLAYER_AMMO_MAX
        pygame.draw.rect(self.screen, self.COLOR_UI_BAR_BG, (10, 35, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_UI_AMMO, (10, 35, int(bar_width * ammo_ratio), bar_height))
        ammo_text = self.font_small.render(f"{self.player_ammo}/{self.PLAYER_AMMO_MAX}", True, self.COLOR_UI_TEXT)
        self.screen.blit(ammo_text, (10 + bar_width / 2 - ammo_text.get_width() / 2, 37))

        # Score and Wave
        score_text = self.font_medium.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 15, 10))
        wave_text = self.font_medium.render(f"WAVE: {self.wave}/{self.MAX_WAVES}", True, self.COLOR_UI_TEXT)
        self.screen.blit(wave_text, (self.WIDTH - wave_text.get_width() - 15, 40))
        
        # Reloading indicator
        if self.is_reloading:
            reload_text = self.font_large.render("RELOADING...", True, self.COLOR_UI_AMMO)
            pos = (self.WIDTH / 2 - reload_text.get_width() / 2, self.HEIGHT / 2 - reload_text.get_height() / 2)
            self.screen.blit(reload_text, pos)

        # Game Over / Win message
        if self.game_over:
            message = "YOU SURVIVED" if self.game_won else "GAME OVER"
            color = (0, 255, 100) if self.game_won else (255, 0, 0)
            end_text = self.font_large.render(message, True, color)
            pos = (self.WIDTH / 2 - end_text.get_width() / 2, self.HEIGHT / 2 - end_text.get_height() / 2 - 20)
            self.screen.blit(end_text, pos)
            final_score_text = self.font_medium.render(f"Final Score: {self.score}", True, self.COLOR_UI_TEXT)
            pos_score = (self.WIDTH / 2 - final_score_text.get_width() / 2, self.HEIGHT / 2 + 20)
            self.screen.blit(final_score_text, pos_score)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave,
            "health": self.player_health,
            "ammo": self.player_ammo,
        }

    def _create_particles(self, pos, color, count, speed_mult, life):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3) * speed_mult
            vel = np.array([math.cos(angle), math.sin(angle)]) * speed
            self.particles.append({
                "pos": pos.copy(),
                "vel": vel,
                "life": self.np_random.integers(life // 2, life),
                "color": color,
                "radius": self.np_random.uniform(2, 4)
            })

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