import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T12:47:28.637423
# Source Brief: brief_01320.md
# Brief Index: 1320
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    Gymnasium environment: Rhythmic Dominance.
    Launch musical notes to disrupt enemy formations, manage energy, and upgrade notes.
    """
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "Launch musical notes to disrupt enemy formations, managing energy to unleash special attacks and overcome "
        "escalating waves of foes."
    )
    user_guide = (
        "Use ↑↓ to aim your launcher and ←→ to adjust power. Press space to fire a note and shift to use your special "
        "attack when energy is full."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 1500 # Extended from brief for more gameplay
        self.MAX_WAVES = 20

        # Colors
        self.COLOR_BG = (15, 10, 40)
        self.COLOR_GRID = (30, 20, 60)
        self.COLOR_PLAYER_PLATFORM = (180, 180, 200)
        self.COLOR_PLAYER_NOTE = (0, 191, 255) # DeepSkyBlue
        self.COLOR_ENEMY = (255, 69, 0) # OrangeRed
        self.COLOR_SPECIAL = (255, 215, 0) # Gold
        self.COLOR_HEALTH = (50, 205, 50) # LimeGreen
        self.COLOR_ENERGY = (148, 0, 211) # DarkViolet
        self.COLOR_UI_TEXT = (240, 240, 240)
        self.COLOR_TRAJECTORY = (255, 255, 255, 100) # White with alpha

        # Player state
        self.MAX_HEALTH = 100
        self.MAX_ENERGY = 100
        self.LAUNCH_ANGLE_MIN = math.pi / 6  # 30 degrees
        self.LAUNCH_ANGLE_MAX = math.pi * 5 / 6 # 150 degrees
        self.LAUNCH_POWER_MIN = 5
        self.LAUNCH_POWER_MAX = 15
        self.NOTE_COOLDOWN_FRAMES = 10
        self.SPECIAL_COOLDOWN_FRAMES = 60
        
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
        self.font_main = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        
        # --- State Variables ---
        # These are initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_health = 0
        self.player_energy = 0
        self.launch_angle = 0
        self.launch_power = 0
        self.note_cooldown = 0
        self.special_cooldown = 0
        self.special_attack_active = False
        self.special_attack_radius = 0
        self.wave_number = 0
        self.enemies = []
        self.notes = []
        self.particles = []

        # Initialize state for the first time
        # self.reset() # No need to call reset in init

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.player_health = self.MAX_HEALTH
        self.player_energy = 0
        
        self.launch_angle = math.pi / 2  # Start pointing straight up
        self.launch_power = (self.LAUNCH_POWER_MIN + self.LAUNCH_POWER_MAX) / 2
        
        self.note_cooldown = 0
        self.special_cooldown = 0
        self.special_attack_active = False
        
        self.wave_number = 1
        self.enemies = []
        self.notes = []
        self.particles = []
        
        self._spawn_wave()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0.0
        
        # --- 1. Handle Actions ---
        movement, space_pressed, shift_pressed = action[0], action[1] == 1, action[2] == 1
        
        # Adjust aim
        if movement == 1: # Up -> Increase Angle
            self.launch_angle += 0.05
        elif movement == 2: # Down -> Decrease Angle
            self.launch_angle -= 0.05
        elif movement == 3: # Left -> Decrease Power
            self.launch_power -= 0.2
        elif movement == 4: # Right -> Increase Power
            self.launch_power += 0.2
        
        self.launch_angle = np.clip(self.launch_angle, self.LAUNCH_ANGLE_MIN, self.LAUNCH_ANGLE_MAX)
        self.launch_power = np.clip(self.launch_power, self.LAUNCH_POWER_MIN, self.LAUNCH_POWER_MAX)
        
        # Launch note
        if space_pressed and self.note_cooldown <= 0:
            self._launch_note()
            # sfx: launch_note.wav
            self.note_cooldown = self.NOTE_COOLDOWN_FRAMES
            
        # Activate special
        if shift_pressed and self.special_cooldown <= 0 and self.player_energy >= self.MAX_ENERGY:
            self.special_attack_active = True
            self.special_attack_radius = 0
            self.player_energy = 0
            self.special_cooldown = self.SPECIAL_COOLDOWN_FRAMES
            # sfx: special_activation.wav

        # --- 2. Update Game State ---
        self._update_cooldowns()
        reward += self._update_notes()
        reward += self._update_enemies()
        self._update_particles()
        reward += self._update_special_attack()

        # --- 3. Check for Wave Completion ---
        if not self.enemies and not self.game_over:
            self.wave_number += 1
            if self.wave_number > self.MAX_WAVES:
                self.game_over = True
                reward += 100 # Win bonus
            else:
                self._spawn_wave()
                reward += 5 # Wave clear bonus
                # sfx: wave_clear.wav

        # --- 4. Check Termination Conditions ---
        terminated = self.game_over or self.steps >= self.MAX_STEPS
        if self.player_health <= 0 and not self.game_over:
            self.game_over = True
            terminated = True
            reward -= 100 # Lose penalty
            # sfx: game_over.wav
        
        truncated = self.steps >= self.MAX_STEPS

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    # --- Update Helpers ---
    def _update_cooldowns(self):
        if self.note_cooldown > 0:
            self.note_cooldown -= 1
        if self.special_cooldown > 0:
            self.special_cooldown -= 1

    def _update_notes(self):
        reward = 0
        for note in self.notes[:]:
            note['pos'] += note['vel']
            
            # Note-Enemy collision
            for enemy in self.enemies[:]:
                if note['pos'].distance_to(enemy['pos']) < enemy['size']:
                    # sfx: enemy_hit.wav
                    self._create_particles(note['pos'], self.COLOR_ENEMY, 20)
                    enemy['health'] -= self._get_note_damage()
                    reward += 0.1 # Hit enemy
                    self.player_energy = min(self.MAX_ENERGY, self.player_energy + 5)
                    
                    if enemy['health'] <= 0:
                        reward += 1.0 # Destroy enemy
                        self.score += 10 * self.wave_number
                        self._create_particles(enemy['pos'], self.COLOR_ENEMY, 40, 5)
                        self.enemies.remove(enemy)
                    
                    if note in self.notes:
                        self.notes.remove(note)
                    break # Note can only hit one enemy
            
            # Out of bounds check
            if not self.screen.get_rect().collidepoint(int(note['pos'].x), int(note['pos'].y)):
                if note in self.notes:
                    self.notes.remove(note)
                    reward -= 0.1 # Missed shot
        return reward

    def _update_enemies(self):
        for enemy in self.enemies[:]:
            # Move enemy
            wave_speed_mod = 1 + (self.wave_number // 5) * 0.05
            enemy['pos'].y += enemy['speed'] * wave_speed_mod
            enemy['pos'].x += math.sin(self.steps * 0.05 + enemy['offset']) * 1.5
            
            # Pulsating effect
            enemy['pulse'] += 0.1
            enemy['render_size'] = enemy['size'] + math.sin(enemy['pulse']) * 3
            
            # Reached bottom
            if enemy['pos'].y > self.HEIGHT - 20:
                self.player_health -= 20
                self._create_particles(enemy['pos'], self.COLOR_PLAYER_PLATFORM, 30)
                self.enemies.remove(enemy)
                # sfx: player_damage.wav
        return 0 # Reward is handled by termination condition

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _update_special_attack(self):
        reward = 0
        if self.special_attack_active:
            self.special_attack_radius += 15
            
            # Damage enemies in radius
            hit_count = 0
            center = pygame.Vector2(self.WIDTH / 2, self.HEIGHT / 2)
            for enemy in self.enemies[:]:
                if center.distance_to(enemy['pos']) < self.special_attack_radius and not enemy.get('hit_by_special', False):
                    # sfx: special_hit.wav
                    enemy['health'] -= 150 # High damage
                    enemy['hit_by_special'] = True
                    hit_count += 1
                    self._create_particles(enemy['pos'], self.COLOR_SPECIAL, 25)
                    if enemy['health'] <= 0:
                        reward += 1.0 # Destroy enemy
                        self.score += 10 * self.wave_number
                        self._create_particles(enemy['pos'], self.COLOR_SPECIAL, 50, 7)
                        self.enemies.remove(enemy)

            if hit_count >= 2:
                reward += 2.0 # Special attack bonus

            if self.special_attack_radius > self.WIDTH:
                self.special_attack_active = False
        return reward
        
    # --- Spawning and Creation ---
    def _spawn_wave(self):
        num_enemies = min(3 + self.wave_number, 10)
        wave_health_mod = 1 + (self.wave_number // 5) * 0.1
        
        for i in range(num_enemies):
            self.enemies.append({
                'pos': pygame.Vector2(random.uniform(50, self.WIDTH - 50), -30 - i * 40),
                'speed': 0.5 + random.uniform(-0.1, 0.1),
                'health': 50 * wave_health_mod,
                'max_health': 50 * wave_health_mod,
                'size': 15,
                'render_size': 15,
                'pulse': random.uniform(0, math.pi * 2),
                'offset': random.uniform(0, math.pi * 2)
            })

    def _launch_note(self):
        vel_x = math.cos(self.launch_angle) * self.launch_power
        vel_y = -math.sin(self.launch_angle) * self.launch_power
        
        self.notes.append({
            'pos': pygame.Vector2(self.WIDTH / 2, self.HEIGHT - 40),
            'vel': pygame.Vector2(vel_x, vel_y)
        })

    def _create_particles(self, pos, color, count, speed_mult=1.0):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 3) * speed_mult
            vel = pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed)
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'life': random.randint(10, 20),
                'color': color,
                'size': random.randint(1, 3)
            })

    # --- Getters ---
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "health": self.player_health,
            "energy": self.player_energy,
            "wave": self.wave_number,
        }
    
    def _get_note_damage(self):
        # Damage increases every 10 waves
        upgrade_level = self.wave_number // 10
        return 30 + upgrade_level * 10

    # --- Rendering ---
    def _render_game(self):
        # Background Grid
        for i in range(0, self.WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (i, 0), (i, self.HEIGHT), 1)
        for i in range(0, self.HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, i), (self.WIDTH, i), 1)
            
        # Launch Platform
        platform_rect = pygame.Rect(self.WIDTH / 2 - 40, self.HEIGHT - 40, 80, 20)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER_PLATFORM, platform_rect, border_radius=5)
        
        # Trajectory Line
        if not self.game_over:
            start_pos = (self.WIDTH / 2, self.HEIGHT - 40)
            end_x = start_pos[0] + math.cos(self.launch_angle) * self.launch_power * 7
            end_y = start_pos[1] - math.sin(self.launch_angle) * self.launch_power * 7
            pygame.draw.aaline(self.screen, self.COLOR_TRAJECTORY, start_pos, (end_x, end_y), 2)
            
        # Particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / 20.0))
            color = (*p['color'], alpha)
            temp_surf = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p['size'], p['size']), p['size'])
            self.screen.blit(temp_surf, (int(p['pos'].x - p['size']), int(p['pos'].y - p['size'])), special_flags=pygame.BLEND_RGBA_ADD)

        # Special Attack
        if self.special_attack_active:
            self._draw_glow_circle(self.screen, self.COLOR_SPECIAL, (self.WIDTH // 2, self.HEIGHT // 2), int(self.special_attack_radius), 20)

        # Notes
        for note in self.notes:
            self._draw_glow_circle(self.screen, self.COLOR_PLAYER_NOTE, (int(note['pos'].x), int(note['pos'].y)), 8, 15)

        # Enemies
        for enemy in self.enemies:
            pos = (int(enemy['pos'].x), int(enemy['pos'].y))
            size = int(enemy['render_size'])
            self._draw_glow_circle(self.screen, self.COLOR_ENEMY, pos, size, 10)
            
            # Enemy health bar
            if enemy['health'] < enemy['max_health']:
                health_pct = max(0, enemy['health'] / enemy['max_health'])
                bar_width = size * 2
                bar_height = 4
                bar_x = pos[0] - size
                bar_y = pos[1] - size - 8
                pygame.draw.rect(self.screen, (80,0,0), (bar_x, bar_y, bar_width, bar_height))
                pygame.draw.rect(self.screen, self.COLOR_HEALTH, (bar_x, bar_y, bar_width * health_pct, bar_height))

    def _render_ui(self):
        # Health Bar
        health_pct = max(0, self.player_health / self.MAX_HEALTH)
        pygame.draw.rect(self.screen, (80,0,0), (10, 10, 200, 20))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH, (10, 10, 200 * health_pct, 20))
        health_text = self.font_small.render(f"HEALTH", True, self.COLOR_UI_TEXT)
        self.screen.blit(health_text, (15, 12))
        
        # Energy Bar
        energy_pct = max(0, self.player_energy / self.MAX_ENERGY)
        pygame.draw.rect(self.screen, (40,0,60), (self.WIDTH - 210, 10, 200, 20))
        pygame.draw.rect(self.screen, self.COLOR_ENERGY, (self.WIDTH - 210, 10, 200 * energy_pct, 20))
        energy_text = self.font_small.render(f"ENERGY", True, self.COLOR_UI_TEXT)
        self.screen.blit(energy_text, (self.WIDTH - 205, 12))

        # Score
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.WIDTH / 2 - score_text.get_width() / 2, 10))
        
        # Wave Number
        wave_text = self.font_main.render(f"WAVE: {self.wave_number}", True, self.COLOR_UI_TEXT)
        self.screen.blit(wave_text, (self.WIDTH / 2 - wave_text.get_width() / 2, self.HEIGHT - 35))
        
        # Upgrade Level
        upgrade_level = self.wave_number // 10
        upgrade_text = self.font_small.render(f"NOTE LV: {upgrade_level + 1}", True, self.COLOR_UI_TEXT)
        self.screen.blit(upgrade_text, (self.WIDTH - upgrade_text.get_width() - 15, self.HEIGHT - 30))

        # Game Over Text
        if self.game_over:
            result_text_str = "VICTORY!" if self.player_health > 0 else "GAME OVER"
            result_text = self.font_main.render(result_text_str, True, self.COLOR_SPECIAL if self.player_health > 0 else self.COLOR_ENEMY)
            text_rect = result_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(result_text, text_rect)

    def _draw_glow_circle(self, surface, color, pos, radius, glow_size):
        """Draws a circle with a soft glow effect."""
        for i in range(glow_size, 0, -2):
            alpha = int(100 * (1 - (i / glow_size)))
            s = pygame.Surface((radius * 2 + i * 2, radius * 2 + i * 2), pygame.SRCALPHA)
            pygame.draw.circle(s, (*color, alpha), (s.get_width() // 2, s.get_height() // 2), radius + i)
            surface.blit(s, (pos[0] - s.get_width() // 2, pos[1] - s.get_height() // 2), special_flags=pygame.BLEND_RGBA_ADD)
        pygame.gfxdraw.aacircle(surface, pos[0], pos[1], radius, color)
        pygame.gfxdraw.filled_circle(surface, pos[0], pos[1], radius, color)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
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
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually
    # To run, you will need to unset the dummy video driver
    # comment out the line `os.environ.setdefault("SDL_VIDEODRIVER", "dummy")`
    # and then run `python your_file_name.py`
    
    # For headed mode, pygame.display must be initialized.
    # We do it here, after checking if the script is run directly.
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
    
    pygame.display.init()
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Rhythmic Dominance")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement = 0 # No-op
        space = 0
        shift = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_w] or keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_s] or keys[pygame.K_DOWN]:
            movement = 2
        elif keys[pygame.K_a] or keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            movement = 4
        
        if keys[pygame.K_SPACE]:
            space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift = 1
            
        action = [movement, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            pygame.time.wait(3000) # Pause for 3 seconds
            obs, info = env.reset()
            total_reward = 0

        clock.tick(env.FPS)
        
    env.close()