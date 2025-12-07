
# Generated: 2025-08-27T12:56:31.159733
# Source Brief: brief_00208.md
# Brief Index: 208

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame


# Set the SDL_VIDEODRIVER to "dummy" to run headless
os.environ["SDL_VIDEODRIVER"] = "dummy"

class GameEnv(gym.Env):
    """
    A top-down arcade shooter where the player controls a robot to fight waves of enemies.
    The game prioritizes visual feedback and a fast-paced "game feel".
    """
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrow keys to move the robot. Hold Space to fire your weapon. "
        "Your robot aims in the direction it last moved."
    )

    game_description = (
        "A fast-paced, top-down arcade shooter. Pilot your robot in a deadly arena, "
        "blasting through waves of increasingly difficult enemies to achieve the highest score."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAX_STEPS = 5000
        self.PLAYER_SIZE = 20
        self.PLAYER_SPEED = 4.0
        self.PLAYER_MAX_HEALTH = 100
        self.ENEMY_SIZE = 18
        self.PROJECTILE_SPEED = 10.0
        self.PROJECTILE_DAMAGE = 25
        self.PROJECTILE_COOLDOWN = 5  # frames
        self.ENEMY_CONTACT_DAMAGE = 1

        # Colors
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_PLAYER = (0, 150, 255)
        self.COLOR_PLAYER_OUTLINE = (100, 200, 255)
        self.COLOR_ENEMY = (255, 50, 50)
        self.COLOR_ENEMY_OUTLINE = (255, 150, 150)
        self.COLOR_PROJECTILE = (255, 255, 0)
        self.COLOR_HEALTH_BAR = (0, 255, 0)
        self.COLOR_HEALTH_BAR_BG = (100, 0, 0)
        self.COLOR_ARENA = (50, 50, 70)
        self.COLOR_SPAWN = (100, 20, 20)
        self.COLOR_TEXT = (255, 255, 255)
        
        # Gymnasium spaces
        self.observation_space = Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup for headless rendering
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_game_over = pygame.font.SysFont("impact", 60)
        
        # State variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.wave_number = 0
        self.player_pos = None
        self.player_health = 0
        self.player_last_move_dir = None
        self.projectile_cooldown_timer = 0
        self.enemies = []
        self.projectiles = []
        self.particles = []
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.wave_number = 1
        
        self.player_pos = np.array([self.WIDTH / 2, self.HEIGHT / 2], dtype=np.float32)
        self.player_health = self.PLAYER_MAX_HEALTH
        self.player_last_move_dir = np.array([0, -1], dtype=np.float32) # Start aiming up
        self.projectile_cooldown_timer = 0
        
        self.enemies = []
        self.projectiles = []
        self.particles = []
        
        self._spawn_wave()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        self.steps += 1
        
        # --- Player Input and Movement ---
        self._handle_input(action)
        
        # --- Update Game Objects ---
        reward += self._update_projectiles()
        enemy_updates = self._update_enemies()
        reward += enemy_updates['reward']
        
        # Player takes damage from enemy contact
        self.player_health -= enemy_updates['contact_damage']
        if enemy_updates['contact_damage'] > 0:
            # sfx: player_damage.wav
            self._create_particles(self.player_pos, 5, self.COLOR_PLAYER, 0.5)

        # --- Update Particles ---
        self._update_particles()
        
        # --- Check Game State ---
        if not self.enemies:
            # sfx: wave_clear.wav
            self.score += 100
            reward += 100
            self.wave_number += 1
            self._spawn_wave()

        if self.player_health <= 0:
            self.player_health = 0
            self.game_over = True
            reward -= 100 # Death penalty
        
        terminated = self.game_over or self.steps >= self.MAX_STEPS
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, action):
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        
        player_vel = np.array([0, 0], dtype=np.float32)
        if movement == 1: player_vel[1] = -1
        elif movement == 2: player_vel[1] = 1
        elif movement == 3: player_vel[0] = -1
        elif movement == 4: player_vel[0] = 1

        if np.linalg.norm(player_vel) > 0:
            player_vel /= np.linalg.norm(player_vel)
            self.player_last_move_dir = player_vel.copy()

        self.player_pos += player_vel * self.PLAYER_SPEED
        self.player_pos[0] = np.clip(self.player_pos[0], self.PLAYER_SIZE / 2, self.WIDTH - self.PLAYER_SIZE / 2)
        self.player_pos[1] = np.clip(self.player_pos[1], self.PLAYER_SIZE / 2, self.HEIGHT - self.PLAYER_SIZE / 2)

        if self.projectile_cooldown_timer > 0:
            self.projectile_cooldown_timer -= 1

        if space_held and self.projectile_cooldown_timer == 0:
            # sfx: player_shoot.wav
            self.projectile_cooldown_timer = self.PROJECTILE_COOLDOWN
            proj_pos = self.player_pos + self.player_last_move_dir * (self.PLAYER_SIZE / 2)
            self.projectiles.append({"pos": proj_pos, "vel": self.player_last_move_dir * self.PROJECTILE_SPEED})
            self._create_particles(proj_pos, 5, self.COLOR_PROJECTILE, 1.0)

    def _update_projectiles(self):
        reward = 0
        projectiles_to_keep = []
        for p in self.projectiles:
            p["pos"] += p["vel"]
            
            if not (0 < p["pos"][0] < self.WIDTH and 0 < p["pos"][1] < self.HEIGHT):
                reward -= 0.02  # Miss penalty
                continue

            hit_enemy = False
            for e in self.enemies:
                if np.linalg.norm(p["pos"] - e["pos"]) < (self.ENEMY_SIZE / 2):
                    # sfx: enemy_hit.wav
                    e["health"] -= self.PROJECTILE_DAMAGE
                    reward += 0.1  # Hit reward
                    self._create_particles(p["pos"], 10, self.COLOR_ENEMY, 1.5)
                    hit_enemy = True
                    break
            
            if not hit_enemy:
                projectiles_to_keep.append(p)
        
        self.projectiles = projectiles_to_keep
        return reward

    def _update_enemies(self):
        reward = 0
        contact_damage = 0
        enemies_to_keep = []
        
        for e in self.enemies:
            if e["health"] <= 0:
                # sfx: enemy_explode.wav
                self.score += 10
                reward += 1.0  # Defeat reward
                self._create_particles(e["pos"], 30, self.COLOR_ENEMY_OUTLINE, 3.0)
                continue
            
            direction = self.player_pos - e["pos"]
            dist = np.linalg.norm(direction)
            if dist > 0:
                direction /= dist
            e["pos"] += direction * e["speed"]
            
            if np.linalg.norm(self.player_pos - e["pos"]) < (self.PLAYER_SIZE / 2 + self.ENEMY_SIZE / 2):
                contact_damage += self.ENEMY_CONTACT_DAMAGE
                e["health"] -= 2 # Enemy also takes minor damage from collision
            
            enemies_to_keep.append(e)

        self.enemies = enemies_to_keep
        return {"reward": reward, "contact_damage": contact_damage}

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1
            p['radius'] *= 0.95

    def _spawn_wave(self):
        num_enemies = min(10, 3 + self.wave_number)
        enemy_speed = 1.0 + (self.wave_number - 1) * 0.2
        enemy_health = 20 + math.floor((self.wave_number - 1) / 2) * 10

        self.enemies.clear()
        for _ in range(num_enemies):
            side = self.np_random.integers(4)
            if side == 0: pos = [self.np_random.uniform(0, self.WIDTH), -self.ENEMY_SIZE]
            elif side == 1: pos = [self.np_random.uniform(0, self.WIDTH), self.HEIGHT + self.ENEMY_SIZE]
            elif side == 2: pos = [-self.ENEMY_SIZE, self.np_random.uniform(0, self.HEIGHT)]
            else: pos = [self.WIDTH + self.ENEMY_SIZE, self.np_random.uniform(0, self.HEIGHT)]

            self.enemies.append({
                "pos": np.array(pos, dtype=np.float32),
                "health": enemy_health, "max_health": enemy_health, "speed": enemy_speed,
            })

    def _create_particles(self, pos, num_particles, color, speed_multiplier=1.0):
        for _ in range(num_particles):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3) * speed_multiplier
            vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed], dtype=np.float32)
            self.particles.append({
                "pos": pos.copy(), "vel": vel,
                "radius": self.np_random.uniform(2, 5), "life": self.np_random.integers(10, 20), "color": color
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Arena and spawn indicators
        pygame.draw.rect(self.screen, self.COLOR_ARENA, (0, 0, self.WIDTH, self.HEIGHT), 2)
        for i in range(5):
            pygame.gfxdraw.filled_circle(self.screen, 100 + i * 110, 5, 3, self.COLOR_SPAWN)
            pygame.gfxdraw.filled_circle(self.screen, 100 + i * 110, self.HEIGHT - 5, 3, self.COLOR_SPAWN)

        # Particles
        for p in self.particles:
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            radius = int(p['radius'])
            if radius > 0:
                alpha = max(0, min(255, int(p['life'] * 12)))
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, p['color'] + (alpha,))
        
        # Projectiles
        for p in self.projectiles:
            start_pos = (int(p['pos'][0]), int(p['pos'][1]))
            end_pos = (int(p['pos'][0] - p['vel'][0]), int(p['pos'][1] - p['vel'][1]))
            pygame.draw.line(self.screen, self.COLOR_PROJECTILE, start_pos, end_pos, 3)

        # Enemies
        for e in self.enemies:
            direction_to_player = self.player_pos - e["pos"]
            angle = math.atan2(direction_to_player[1], direction_to_player[0])
            points = [(int(e["pos"][0] + math.cos(angle + i * 2.2) * self.ENEMY_SIZE),
                       int(e["pos"][1] + math.sin(angle + i * 2.2) * self.ENEMY_SIZE)) for i in range(3)]
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_ENEMY_OUTLINE)
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_ENEMY)
            self._render_health_bar(e['pos'], e['health'], e['max_health'], self.ENEMY_SIZE)

        # Player
        px, py = int(self.player_pos[0]), int(self.player_pos[1])
        player_rect = pygame.Rect(px - self.PLAYER_SIZE / 2, py - self.PLAYER_SIZE / 2, self.PLAYER_SIZE, self.PLAYER_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER_OUTLINE, player_rect, 2)
        aim_end = (px + self.player_last_move_dir[0] * 15, py + self.player_last_move_dir[1] * 15)
        pygame.draw.line(self.screen, self.COLOR_PLAYER_OUTLINE, (px, py), (int(aim_end[0]), int(aim_end[1])), 2)
        self._render_health_bar(self.player_pos, self.player_health, self.PLAYER_MAX_HEALTH, self.PLAYER_SIZE)

    def _render_health_bar(self, pos, current_health, max_health, entity_size):
        bar_width, bar_height = entity_size * 1.5, 5
        x, y = pos[0] - bar_width / 2, pos[1] - entity_size / 2 - bar_height - 5
        health_ratio = max(0, current_health / max_health)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (int(x), int(y), int(bar_width), bar_height))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, (int(x), int(y), int(bar_width * health_ratio), bar_height))

    def _render_ui(self):
        score_text = self.font_ui.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        health_text = self.font_ui.render(f"Health: {int(self.player_health)}", True, self.COLOR_TEXT)
        self.screen.blit(health_text, health_text.get_rect(topright=(self.WIDTH - 10, 10)))

        wave_text = self.font_ui.render(f"Wave: {self.wave_number}", True, self.COLOR_TEXT)
        self.screen.blit(wave_text, wave_text.get_rect(midbottom=(self.WIDTH / 2, self.HEIGHT - 10)))

        if self.game_over:
            go_text = self.font_game_over.render("GAME OVER", True, self.COLOR_ENEMY)
            self.screen.blit(go_text, go_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2)))

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "wave": self.wave_number}

    def close(self):
        pygame.font.quit()
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        # Test observation space after reset
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc is False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")