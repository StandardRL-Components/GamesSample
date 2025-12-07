
# Generated: 2025-08-27T13:42:44.496283
# Source Brief: brief_00458.md
# Brief Index: 458

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move and aim. Space to shoot. Shift to reload."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Survive hordes of procedurally generated zombies in a top-down shooter for 60 seconds."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 60 * self.FPS

        # Colors
        self.COLOR_BG = (20, 20, 30)
        self.COLOR_PLAYER = (0, 200, 255)
        self.COLOR_PLAYER_GLOW = (0, 200, 255, 50)
        self.COLOR_ZOMBIE = (60, 180, 70)
        self.COLOR_PROJECTILE = (255, 255, 100)
        self.COLOR_PARTICLE_HIT = (255, 255, 255)
        self.COLOR_PARTICLE_DEATH = (255, 100, 50)
        self.COLOR_UI_TEXT = (230, 230, 230)
        self.COLOR_UI_HEALTH_BG = (100, 0, 0)
        self.COLOR_UI_HEALTH = (255, 0, 0)
        self.COLOR_UI_AMMO = (200, 200, 0)
        self.COLOR_RELOAD = (255, 165, 0)

        # Player settings
        self.PLAYER_SIZE = 12
        self.PLAYER_SPEED = 4
        self.PLAYER_MAX_HEALTH = 50
        self.PLAYER_MAX_AMMO = 10
        self.SHOOT_COOLDOWN = 5  # frames
        self.RELOAD_TIME = 45 # frames (1.5 seconds)

        # Zombie settings
        self.ZOMBIE_SIZE = 10
        self.ZOMBIE_HEALTH = 10
        self.ZOMBIE_DAMAGE = 1
        self.INITIAL_ZOMBIE_SPAWN_RATE = 1.0  # per second
        self.ZOMBIE_SPAWN_RATE_INCREASE = 0.01 # per second
        self.INITIAL_ZOMBIE_SPEED = 0.5
        self.ZOMBIE_SPEED_INCREASE = 0.005 # per second

        # Projectile settings
        self.PROJECTILE_SPEED = 12
        self.PROJECTILE_DAMAGE = 10

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 28)
        self.font_score = pygame.font.Font(None, 40)
        self.font_reload = pygame.font.Font(None, 32)

        # --- State Variables ---
        self.player_pos = None
        self.player_health = None
        self.player_aim_angle = None
        self.player_last_move_dir = None
        self.player_ammo = None
        self.shoot_cooldown_timer = None
        self.reloading_timer = None

        self.zombies = None
        self.projectiles = None
        self.particles = None
        
        self.zombie_spawn_timer = None
        
        self.screen_shake_timer = None

        self.steps = None
        self.score = None
        self.game_over = None
        self.np_random = None

        # --- Final Validation ---
        if render_mode == "rgb_array": # Avoid issues in environments without video drivers
            try:
                import os
                os.environ["SDL_VIDEODRIVER"] = "dummy"
            except ImportError:
                print("Warning: could not set SDL_VIDEODRIVER to dummy.")

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)

        self.player_pos = np.array([self.WIDTH / 2, self.HEIGHT / 2], dtype=np.float32)
        self.player_health = self.PLAYER_MAX_HEALTH
        self.player_aim_angle = -math.pi / 2  # Start facing up
        self.player_last_move_dir = np.array([0, -1], dtype=np.float32)
        self.player_ammo = self.PLAYER_MAX_AMMO
        self.shoot_cooldown_timer = 0
        self.reloading_timer = 0
        
        self.zombies = []
        self.projectiles = []
        self.particles = deque()

        self.zombie_spawn_timer = 0
        self.screen_shake_timer = 0

        self.steps = 0
        self.score = 0
        self.game_over = False

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
        
        if self.auto_advance:
            self.clock.tick(self.FPS)

        self.steps += 1
        reward = 0

        # --- Handle Input & Player Logic ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # Movement and Aiming
        move_vec = np.array([0, 0], dtype=np.float32)
        if movement == 1: move_vec[1] = -1 # Up
        elif movement == 2: move_vec[1] = 1 # Down
        elif movement == 3: move_vec[0] = -1 # Left
        elif movement == 4: move_vec[0] = 1 # Right

        if np.any(move_vec):
            self.player_pos += move_vec * self.PLAYER_SPEED
            self.player_pos[0] = np.clip(self.player_pos[0], self.PLAYER_SIZE, self.WIDTH - self.PLAYER_SIZE)
            self.player_pos[1] = np.clip(self.player_pos[1], self.PLAYER_SIZE, self.HEIGHT - self.PLAYER_SIZE)
            self.player_aim_angle = math.atan2(move_vec[1], move_vec[0])
            self.player_last_move_dir = move_vec
        
        # Timers
        if self.shoot_cooldown_timer > 0: self.shoot_cooldown_timer -= 1
        if self.reloading_timer > 0: self.reloading_timer -= 1
        if self.screen_shake_timer > 0: self.screen_shake_timer -= 1

        # Reloading
        if shift_held and self.player_ammo < self.PLAYER_MAX_AMMO and self.reloading_timer == 0:
            self.reloading_timer = self.RELOAD_TIME
            # sfx: reload_start
        if self.reloading_timer == 1: # Just finished
            self.player_ammo = self.PLAYER_MAX_AMMO
            # sfx: reload_complete

        # Shooting
        if space_held and self.shoot_cooldown_timer == 0 and self.player_ammo > 0 and self.reloading_timer == 0:
            self._spawn_projectile()
            self.player_ammo -= 1
            self.shoot_cooldown_timer = self.SHOOT_COOLDOWN
            # sfx: shoot_laser

        # --- Update Game State ---
        reward += self._update_projectiles()
        reward += self._update_zombies()
        self._update_particles()
        self._spawn_zombies()

        # --- Termination ---
        terminated = self.player_health <= 0 or self.steps >= self.MAX_STEPS
        if terminated:
            self.game_over = True
            if self.player_health <= 0:
                reward -= 100
                # sfx: player_death
            else: # Survived
                reward += 100
                # sfx: victory_fanfare

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _update_projectiles(self):
        reward = 0
        projectiles_to_remove = []
        zombies_to_remove = []
        
        for proj in self.projectiles:
            proj['pos'] += proj['vel']
            if not (0 < proj['pos'][0] < self.WIDTH and 0 < proj['pos'][1] < self.HEIGHT):
                projectiles_to_remove.append(proj)
                reward -= 0.01 # Miss penalty
                continue
            
            for zombie in self.zombies:
                if np.linalg.norm(proj['pos'] - zombie['pos']) < self.ZOMBIE_SIZE:
                    reward += 0.1 # Hit reward
                    zombie['health'] -= self.PROJECTILE_DAMAGE
                    self._spawn_particles(proj['pos'], 10, self.COLOR_PARTICLE_HIT, 1, 3, 2, 15)
                    projectiles_to_remove.append(proj)
                    # sfx: zombie_hit
                    if zombie['health'] <= 0 and zombie not in zombies_to_remove:
                        reward += 1.0 # Kill reward
                        self.score += 1
                        zombies_to_remove.append(zombie)
                        self._spawn_particles(zombie['pos'], 30, self.COLOR_PARTICLE_DEATH, 2, 5, 1, 25)
                        # sfx: zombie_death_explosion
                    break
        
        self.projectiles = [p for p in self.projectiles if p not in projectiles_to_remove]
        self.zombies = [z for z in self.zombies if z not in zombies_to_remove]
        return reward

    def _update_zombies(self):
        reward = 0
        zombies_to_remove = []
        for zombie in self.zombies:
            direction = self.player_pos - zombie['pos']
            dist = np.linalg.norm(direction)
            if dist > 1:
                direction /= dist
            zombie['pos'] += direction * zombie['speed']

            if np.linalg.norm(self.player_pos - zombie['pos']) < self.PLAYER_SIZE + self.ZOMBIE_SIZE:
                self.player_health -= self.ZOMBIE_DAMAGE
                self.screen_shake_timer = 5
                zombies_to_remove.append(zombie)
                self._spawn_particles(self.player_pos, 15, self.COLOR_UI_HEALTH, 1, 3, 2, 20)
                # sfx: player_hurt
        
        self.zombies = [z for z in self.zombies if z not in zombies_to_remove]
        return reward

    def _update_particles(self):
        particles_to_remove = 0
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1
            if p['life'] <= 0:
                particles_to_remove += 1
        for _ in range(particles_to_remove):
            self.particles.popleft()

    def _spawn_zombies(self):
        self.zombie_spawn_timer -= 1
        if self.zombie_spawn_timer <= 0:
            elapsed_seconds = self.steps / self.FPS
            current_spawn_rate = self.INITIAL_ZOMBIE_SPAWN_RATE + elapsed_seconds * self.ZOMBIE_SPAWN_RATE_INCREASE
            
            side = self.np_random.integers(4)
            if side == 0: # Top
                pos = np.array([self.np_random.uniform(0, self.WIDTH), -self.ZOMBIE_SIZE], dtype=np.float32)
            elif side == 1: # Bottom
                pos = np.array([self.np_random.uniform(0, self.WIDTH), self.HEIGHT + self.ZOMBIE_SIZE], dtype=np.float32)
            elif side == 2: # Left
                pos = np.array([-self.ZOMBIE_SIZE, self.np_random.uniform(0, self.HEIGHT)], dtype=np.float32)
            else: # Right
                pos = np.array([self.WIDTH + self.ZOMBIE_SIZE, self.np_random.uniform(0, self.HEIGHT)], dtype=np.float32)
            
            current_speed = self.INITIAL_ZOMBIE_SPEED + elapsed_seconds * self.ZOMBIE_SPEED_INCREASE
            
            self.zombies.append({
                'pos': pos,
                'health': self.ZOMBIE_HEALTH,
                'speed': current_speed,
                'color': (
                    self.np_random.integers(40, 80),
                    self.np_random.integers(160, 200),
                    self.np_random.integers(60, 90)
                )
            })
            self.zombie_spawn_timer = self.FPS / current_spawn_rate

    def _spawn_projectile(self):
        vel = np.array([math.cos(self.player_aim_angle), math.sin(self.player_aim_angle)]) * self.PROJECTILE_SPEED
        # Muzzle flash
        self._spawn_particles(self.player_pos + vel/2, 8, (255,255,255), 1, 2, 1, 5)
        self.projectiles.append({'pos': self.player_pos.copy(), 'vel': vel})
        
    def _spawn_particles(self, pos, count, color, min_speed, max_speed, min_life, max_life):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(min_speed, max_speed)
            vel = np.array([math.cos(angle), math.sin(angle)]) * speed
            life = self.np_random.integers(min_life, max_life + 1)
            self.particles.append({'pos': pos.copy(), 'vel': vel, 'life': life, 'color': color})

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        
        offset = (0, 0)
        if self.screen_shake_timer > 0:
            offset = (self.np_random.integers(-3, 4), self.np_random.integers(-3, 4))

        self._render_game(offset)
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self, offset):
        # Particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / 10))
            if alpha > 0:
                pos = (int(p['pos'][0] + offset[0]), int(p['pos'][1] + offset[1]))
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], max(0, int(p['life'] / 4)), (*p['color'], alpha))

        # Zombies
        for z in self.zombies:
            pos = (int(z['pos'][0] + offset[0]), int(z['pos'][1] + offset[1]))
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.ZOMBIE_SIZE, z['color'])
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.ZOMBIE_SIZE, z['color'])

        # Projectiles
        for p in self.projectiles:
            start_pos = (int(p['pos'][0] + offset[0]), int(p['pos'][1] + offset[1]))
            end_pos_vec = p['pos'] - p['vel']
            end_pos = (int(end_pos_vec[0] + offset[0]), int(end_pos_vec[1] + offset[1]))
            pygame.draw.aaline(self.screen, self.COLOR_PROJECTILE, start_pos, end_pos, 2)

        # Player
        player_x, player_y = int(self.player_pos[0] + offset[0]), int(self.player_pos[1] + offset[1])
        # Glow
        pygame.gfxdraw.filled_circle(self.screen, player_x, player_y, int(self.PLAYER_SIZE * 1.5), self.COLOR_PLAYER_GLOW)
        # Body
        p1 = (player_x + self.PLAYER_SIZE * math.cos(self.player_aim_angle), player_y + self.PLAYER_SIZE * math.sin(self.player_aim_angle))
        p2 = (player_x + self.PLAYER_SIZE * math.cos(self.player_aim_angle + 2.3), player_y + self.PLAYER_SIZE * math.sin(self.player_aim_angle + 2.3))
        p3 = (player_x + self.PLAYER_SIZE * math.cos(self.player_aim_angle - 2.3), player_y + self.PLAYER_SIZE * math.sin(self.player_aim_angle - 2.3))
        pygame.gfxdraw.aapolygon(self.screen, [p1, p2, p3], self.COLOR_PLAYER)
        pygame.gfxdraw.filled_polygon(self.screen, [p1, p2, p3], self.COLOR_PLAYER)

    def _render_ui(self):
        # Health Bar
        health_pct = max(0, self.player_health / self.PLAYER_MAX_HEALTH)
        pygame.draw.rect(self.screen, self.COLOR_UI_HEALTH_BG, (10, 10, 200, 20))
        pygame.draw.rect(self.screen, self.COLOR_UI_HEALTH, (10, 10, int(200 * health_pct), 20))
        health_text = self.font_ui.render(f"HP: {self.player_health}", True, self.COLOR_UI_TEXT)
        self.screen.blit(health_text, (15, 12))

        # Timer
        time_left = max(0, (self.MAX_STEPS - self.steps) / self.FPS)
        timer_text = self.font_ui.render(f"TIME: {time_left:.1f}", True, self.COLOR_UI_TEXT)
        self.screen.blit(timer_text, (self.WIDTH - timer_text.get_width() - 10, 10))

        # Score
        score_text = self.font_score.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.WIDTH/2 - score_text.get_width()/2, self.HEIGHT - 40))

        # Ammo
        ammo_text_str = f"AMMO: {self.player_ammo}/{self.PLAYER_MAX_AMMO}"
        ammo_color = self.COLOR_UI_AMMO
        if self.player_ammo == 0:
            ammo_color = self.COLOR_UI_HEALTH
        ammo_text = self.font_ui.render(ammo_text_str, True, ammo_color)
        self.screen.blit(ammo_text, (10, 35))

        # Reloading indicator
        if self.reloading_timer > 0:
            reload_text = self.font_reload.render("RELOADING...", True, self.COLOR_RELOAD)
            self.screen.blit(reload_text, (self.WIDTH/2 - reload_text.get_width()/2, self.HEIGHT/2 - 50))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "health": self.player_health,
            "ammo": self.player_ammo,
            "time_left": max(0, (self.MAX_STEPS - self.steps) / self.FPS)
        }
        
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation.
        """
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")


if __name__ == '__main__':
    # This block allows you to run the file directly to play the game
    env = GameEnv(render_mode="rgb_array")
    
    # We need a window to display the game
    pygame.display.set_caption(env.game_description)
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    obs, info = env.reset()
    terminated = False
    
    # Game loop for human play
    running = True
    while running:
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # --- Action Mapping for Human ---
        keys = pygame.key.get_pressed()
        movement = 0 # No-op
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4

        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_SHIFT] else 0
        
        action = [movement, space_held, shift_held]

        # --- Step Environment ---
        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
        else:
            # If game over, allow reset with 'R' key
            if keys[pygame.K_r]:
                obs, info = env.reset()
                terminated = False

        # --- Rendering ---
        # The observation is already a rendered frame, so we just display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        if terminated:
            # Display game over message
            font_game_over = pygame.font.Font(None, 72)
            msg = "YOU DIED" if info['health'] <= 0 else "YOU SURVIVED!"
            color = (255, 50, 50) if info['health'] <= 0 else (50, 255, 50)
            
            text_surf = font_game_over.render(msg, True, color)
            text_rect = text_surf.get_rect(center=(env.WIDTH/2, env.HEIGHT/2 - 30))
            screen.blit(text_surf, text_rect)
            
            font_restart = pygame.font.Font(None, 36)
            restart_surf = font_restart.render("Press 'R' to Restart", True, (255, 255, 255))
            restart_rect = restart_surf.get_rect(center=(env.WIDTH/2, env.HEIGHT/2 + 30))
            screen.blit(restart_surf, restart_rect)
        
        pygame.display.flip()

    env.close()