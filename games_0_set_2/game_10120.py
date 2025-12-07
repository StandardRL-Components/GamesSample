import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T10:51:12.841762
# Source Brief: brief_00120.md
# Brief Index: 120
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "Top-down arcade space shooter. Destroy enemy waves, collect energy, and unlock more powerful ships."
    )
    user_guide = (
        "Controls: Use arrow keys to move. Press space to fire and shift to use your ship's special ability."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and Game Constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAX_STEPS = 5000
        self.FPS = 30 # For visual interpolation, logic is per-step

        # EXACT spaces:
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)
        
        # Colors
        self.COLOR_BG = (15, 10, 25)
        self.COLOR_PLAYER = (0, 255, 150)
        self.COLOR_PLAYER_SHIELD = (0, 150, 255, 100)
        self.COLOR_ENEMY_WEAK = (255, 50, 50)
        self.COLOR_ENEMY_STRONG = (255, 150, 50)
        self.COLOR_ENERGY = (50, 150, 255)
        self.COLOR_TEXT = (220, 220, 240)
        self.COLOR_UI_BAR = (40, 30, 60)
        self.COLOR_UI_HEALTH = (0, 255, 150)
        self.COLOR_UI_ENERGY = (50, 150, 255)
        self.COLOR_UI_SPECIAL = (255, 200, 0)
        
        # Ship Configurations
        self.SHIP_SPECS = [
            {'name': 'Scout', 'speed': 6, 'fire_rate': 8, 'health': 100, 'energy': 100, 'special': 'none'},
            {'name': 'Interceptor', 'speed': 8, 'fire_rate': 6, 'health': 80, 'energy': 120, 'special': 'shield'},
            {'name': 'Dreadnought', 'speed': 4, 'fire_rate': 4, 'health': 150, 'energy': 150, 'special': 'rapid_fire'}
        ]
        self.UNLOCK_THRESHOLDS = [100, 250]

        # Persistent state (survives reset)
        self.total_energy_collected = 0
        self.unlocked_ships = [True, False, False]
        self.current_ship_index = 0
        
        # Initialize state variables
        self.player_pos = None
        self.player_vel = None
        self.player_health = None
        self.player_energy = None
        self.player_fire_cooldown = None
        self.player_special_cooldown = None
        self.player_special_active = None
        self.player_special_timer = None
        self.enemies = []
        self.projectiles = []
        self.enemy_projectiles = []
        self.energy_orbs = []
        self.particles = []
        self.stars = []
        
        self._generate_stars(200)
        
        # self.reset() is called by the environment runner, not needed in __init__
        # self.validate_implementation() is a self-check and not part of the final env
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.galaxy_number = 1
        self.difficulty_modifier = 1.0
        self.step_reward = 0

        # Player state
        ship_spec = self.SHIP_SPECS[self.current_ship_index]
        self.player_pos = np.array([self.WIDTH / 2, self.HEIGHT / 2], dtype=np.float32)
        self.player_vel = np.array([0, 0], dtype=np.float32)
        self.player_health = ship_spec['health']
        self.player_max_health = ship_spec['health']
        self.player_energy = 50
        self.player_max_energy = ship_spec['energy']
        self.player_fire_cooldown = 0
        self.player_special_cooldown = 0
        self.player_special_active = False
        self.player_special_timer = 0
        
        # Object lists
        self.enemies.clear()
        self.projectiles.clear()
        self.enemy_projectiles.clear()
        self.energy_orbs.clear()
        self.particles.clear()
        
        self._spawn_enemies_for_galaxy()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            # If the game is over, subsequent steps do nothing but return the final state
            # The terminated/truncated flags should remain True from the step that ended the game.
            # Here we ensure we return a valid 5-tuple.
            terminated = self.player_health <= 0
            truncated = self.steps >= self.MAX_STEPS
            return self._get_observation(), 0, terminated, truncated, self._get_info()

        self.step_reward = 0
        
        # 1. Handle Input
        self._handle_input(action)

        # 2. Update Game State
        self._update_player()
        self._update_projectiles()
        self._update_enemies()
        self._update_particles()
        
        # 3. Handle Collisions
        self._handle_collisions()

        # 4. Cleanup
        self._cleanup_objects()
        
        # 5. Check for galaxy progression
        if not self.enemies:
            self._next_galaxy()

        # 6. Update difficulty
        if self.steps > 0 and self.steps % 500 == 0:
            self.difficulty_modifier += 0.05
        
        self.steps += 1
        
        # 7. Check for termination and truncation
        reward = self.step_reward
        terminated = self.player_health <= 0
        truncated = self.steps >= self.MAX_STEPS
        
        if terminated:
            self.game_over = True
            reward -= 100 # Terminal penalty
            self._spawn_particle_explosion(self.player_pos, self.COLOR_PLAYER, 100, 10)
        elif truncated:
            self.game_over = True
            reward += 100 # Terminal bonus for reaching step limit
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        ship_spec = self.SHIP_SPECS[self.current_ship_index]
        
        # Movement
        move_direction = np.array([0, 0], dtype=np.float32)
        if movement == 1: move_direction[1] = -1 # Up
        elif movement == 2: move_direction[1] = 1 # Down
        elif movement == 3: move_direction[0] = -1 # Left
        elif movement == 4: move_direction[0] = 1 # Right
        
        if np.linalg.norm(move_direction) > 0:
            move_direction /= np.linalg.norm(move_direction)
        
        self.player_vel = move_direction * ship_spec['speed']
        
        # Primary Fire
        if space_held and self.player_fire_cooldown <= 0:
            fire_rate = ship_spec['fire_rate']
            if self.player_special_active and ship_spec['special'] == 'rapid_fire':
                fire_rate //= 2 # Halve cooldown time
                
            self.player_fire_cooldown = fire_rate
            self.projectiles.append({
                'pos': self.player_pos.copy() + np.array([0, -20]),
                'vel': np.array([0, -15]), 'color': self.COLOR_PLAYER, 'size': 4
            })
            # sfx: player_shoot.wav
            
        # Special Ability
        if shift_held and self.player_special_cooldown <= 0 and self.player_energy >= 50:
            if ship_spec['special'] == 'shield':
                self.player_special_active = True
                self.player_special_timer = 150 # 5 seconds at 30fps
                self.player_energy -= 50
                self.player_special_cooldown = 300
                # sfx: shield_on.wav
            elif ship_spec['special'] == 'rapid_fire':
                self.player_special_active = True
                self.player_special_timer = 150 # 5 seconds
                self.player_energy -= 50
                self.player_special_cooldown = 300
                # sfx: rapid_fire_on.wav
    
    def _update_player(self):
        ship_spec = self.SHIP_SPECS[self.current_ship_index]
        
        # Apply velocity
        self.player_pos += self.player_vel
        
        # Friction
        self.player_vel *= 0.8
        
        # Boundary checks
        self.player_pos[0] = np.clip(self.player_pos[0], 15, self.WIDTH - 15)
        self.player_pos[1] = np.clip(self.player_pos[1], 15, self.HEIGHT - 15)
        
        # Cooldowns
        if self.player_fire_cooldown > 0: self.player_fire_cooldown -= 1
        if self.player_special_cooldown > 0: self.player_special_cooldown -= 1
        
        # Special ability timer
        if self.player_special_active:
            self.player_special_timer -= 1
            if self.player_special_timer <= 0:
                self.player_special_active = False
                # sfx: ability_off.wav
        
        # Engine particles
        if np.linalg.norm(self.player_vel) > 0.1:
            for _ in range(2):
                self.particles.append({
                    'pos': self.player_pos.copy() + np.array([random.uniform(-5, 5), 15]),
                    'vel': np.array([random.uniform(-0.5, 0.5), random.uniform(2, 4)]),
                    'life': 15, 'max_life': 15, 'color': self.COLOR_UI_SPECIAL
                })

    def _update_projectiles(self):
        for p in self.projectiles: p['pos'] += p['vel']
        for p in self.enemy_projectiles: p['pos'] += p['vel']

    def _update_enemies(self):
        for enemy in self.enemies:
            # Movement patterns
            if enemy['type'] == 'grunt':
                enemy['pos'][0] += enemy['vel'][0] * self.difficulty_modifier
                if enemy['pos'][0] < 20 or enemy['pos'][0] > self.WIDTH - 20:
                    enemy['vel'][0] *= -1
            elif enemy['type'] == 'striker':
                target_dir = self.player_pos - enemy['pos']
                dist = np.linalg.norm(target_dir)
                if dist > 0: target_dir /= dist
                
                if dist > 150:
                    enemy['pos'] += target_dir * enemy['speed'] * self.difficulty_modifier
                else:
                    enemy['pos'] -= target_dir * enemy['speed'] * 0.5 * self.difficulty_modifier

            # Firing
            enemy['fire_cooldown'] -= 1
            if enemy['fire_cooldown'] <= 0:
                fire_rate = int(enemy['base_fire_rate'] / self.difficulty_modifier)
                enemy['fire_cooldown'] = random.randint(fire_rate, fire_rate + 30)
                
                target_dir = self.player_pos - enemy['pos']
                dist = np.linalg.norm(target_dir)
                if dist > 0: target_dir /= dist
                
                self.enemy_projectiles.append({
                    'pos': enemy['pos'].copy(), 'vel': target_dir * 5,
                    'color': self.COLOR_ENEMY_WEAK, 'size': 3
                })
                # sfx: enemy_shoot.wav
    
    def _update_particles(self):
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1

    def _handle_collisions(self):
        # Player projectiles vs Enemies
        for proj in self.projectiles[:]:
            for enemy in self.enemies[:]:
                if np.linalg.norm(proj['pos'] - enemy['pos']) < enemy['size']:
                    self.projectiles.remove(proj)
                    enemy['health'] -= 25
                    self._spawn_particle_explosion(enemy['pos'], (255, 255, 255), 5, 2)
                    if enemy['health'] <= 0:
                        self.enemies.remove(enemy)
                        self.step_reward += 1.0 # Defeat enemy reward
                        # sfx: enemy_explosion.wav
                        self._spawn_particle_explosion(enemy['pos'], enemy['color'], 30, 4)
                        if random.random() < 0.75: # 75% chance to drop orb
                            self._spawn_energy_orb(enemy['pos'])
                    break
        
        # Player vs Enemy projectiles
        for proj in self.enemy_projectiles[:]:
            if np.linalg.norm(proj['pos'] - self.player_pos) < 15:
                self.enemy_projectiles.remove(proj)
                self._damage_player(10)
                break
        
        # Player vs Enemies
        for enemy in self.enemies[:]:
            if np.linalg.norm(enemy['pos'] - self.player_pos) < 10 + enemy['size']:
                self._damage_player(20)
                self.enemies.remove(enemy)
                self._spawn_particle_explosion(enemy['pos'], enemy['color'], 30, 4)
                break
                
        # Player vs Energy orbs
        for orb in self.energy_orbs[:]:
            if np.linalg.norm(orb['pos'] - self.player_pos) < 20:
                self.energy_orbs.remove(orb)
                self.step_reward += 0.1 # Collect orb reward
                
                old_total = self.total_energy_collected
                self.player_energy = min(self.player_max_energy, self.player_energy + 25)
                self.total_energy_collected += 25
                self._check_unlocks(old_total)
                # sfx: collect_orb.wav
                break

    def _damage_player(self, amount):
        if self.player_special_active and self.SHIP_SPECS[self.current_ship_index]['special'] == 'shield':
            # sfx: shield_hit.wav
            return # Invulnerable
        
        self.player_health -= amount
        self.step_reward -= 0.1 # Damage penalty
        # sfx: player_hit.wav
        self._spawn_particle_explosion(self.player_pos, (255, 255, 255), 10, 3)

    def _cleanup_objects(self):
        self.projectiles = [p for p in self.projectiles if 0 < p['pos'][0] < self.WIDTH and 0 < p['pos'][1] < self.HEIGHT]
        self.enemy_projectiles = [p for p in self.enemy_projectiles if 0 < p['pos'][0] < self.WIDTH and 0 < p['pos'][1] < self.HEIGHT]
        self.particles = [p for p in self.particles if p['life'] > 0]
        
    def _next_galaxy(self):
        self.galaxy_number += 1
        self.step_reward += 5.0 # New galaxy reward
        
        # Reset player state for new galaxy
        self.player_pos = np.array([self.WIDTH / 2, self.HEIGHT / 2], dtype=np.float32)
        self.player_health = self.player_max_health # Full health
        
        self._spawn_enemies_for_galaxy()
        # sfx: galaxy_warp.wav
        self._spawn_particle_explosion(self.player_pos, self.COLOR_PLAYER, 50, 8)

    def _spawn_enemies_for_galaxy(self):
        num_grunts = 2 + self.galaxy_number
        for _ in range(num_grunts):
            self.enemies.append({
                'pos': np.array([random.uniform(50, self.WIDTH-50), random.uniform(50, 150)], dtype=np.float32),
                'vel': np.array([random.choice([-2, 2]), 0], dtype=np.float32),
                'health': 50, 'size': 15, 'type': 'grunt', 'color': self.COLOR_ENEMY_WEAK,
                'fire_cooldown': random.randint(60, 120), 'base_fire_rate': 120
            })
        
        if self.galaxy_number >= 3:
            num_strikers = self.galaxy_number - 2
            for _ in range(num_strikers):
                self.enemies.append({
                    'pos': np.array([random.choice([50, self.WIDTH-50]), random.uniform(50, 200)], dtype=np.float32),
                    'health': 100, 'size': 18, 'type': 'striker', 'color': self.COLOR_ENEMY_STRONG,
                    'speed': 1.5, 'fire_cooldown': random.randint(90, 150), 'base_fire_rate': 90
                })

    def _spawn_energy_orb(self, position):
        self.energy_orbs.append({'pos': position.copy()})
        
    def _spawn_particle_explosion(self, position, color, count, max_speed):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, max_speed)
            vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
            life = random.randint(20, 40)
            self.particles.append({
                'pos': position.copy(), 'vel': vel, 'life': life, 'max_life': life, 'color': color
            })

    def _check_unlocks(self, old_total_energy):
        for i, threshold in enumerate(self.UNLOCK_THRESHOLDS, 1):
            if old_total_energy < threshold <= self.total_energy_collected:
                if not self.unlocked_ships[i]:
                    self.unlocked_ships[i] = True
                    self.step_reward += 10.0 # Unlock reward
                    # In a real game, this would trigger a UI notification
    
    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_background()
        self._render_orbs()
        self._render_projectiles()
        self._render_enemies()
        self._render_player()
        self._render_particles()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for star in self.stars:
            pos = (
                (star['pos'][0] - self.player_pos[0] * star['depth'] * 0.1) % self.WIDTH,
                (star['pos'][1] - self.player_pos[1] * star['depth'] * 0.1) % self.HEIGHT
            )
            pygame.draw.circle(self.screen, star['color'], pos, star['size'])
            
    def _render_player(self):
        pos = self.player_pos.astype(int)
        
        # Glow effect
        glow_radius = 30
        s = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(s, (self.COLOR_PLAYER[0], self.COLOR_PLAYER[1], self.COLOR_PLAYER[2], 50), (glow_radius, glow_radius), glow_radius)
        self.screen.blit(s, (pos[0] - glow_radius, pos[1] - glow_radius))

        # Ship body
        p1 = (pos[0], pos[1] - 15)
        p2 = (pos[0] - 10, pos[1] + 10)
        p3 = (pos[0] + 10, pos[1] + 10)
        pygame.gfxdraw.aatrigon(self.screen, p1[0], p1[1], p2[0], p2[1], p3[0], p3[1], self.COLOR_PLAYER)
        pygame.gfxdraw.filled_trigon(self.screen, p1[0], p1[1], p2[0], p2[1], p3[0], p3[1], self.COLOR_PLAYER)

        # Shield effect
        if self.player_special_active and self.SHIP_SPECS[self.current_ship_index]['special'] == 'shield':
            alpha = int(100 * (self.player_special_timer / 150))
            s = pygame.Surface((40 * 2, 40 * 2), pygame.SRCALPHA)
            pygame.draw.circle(s, self.COLOR_PLAYER_SHIELD, (40, 40), 25, 3)
            self.screen.blit(s, (pos[0] - 40, pos[1] - 40))

    def _render_enemies(self):
        for enemy in self.enemies:
            pos = enemy['pos'].astype(int)
            size = int(enemy['size'])
            if enemy['type'] == 'grunt':
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], size, enemy['color'])
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], size, enemy['color'])
            elif enemy['type'] == 'striker':
                p1 = (pos[0], pos[1] - size)
                p2 = (pos[0] - size, pos[1] + size)
                p3 = (pos[0] + size, pos[1] + size)
                pygame.gfxdraw.aatrigon(self.screen, p1[0], p1[1], p2[0], p2[1], p3[0], p3[1], enemy['color'])
                pygame.gfxdraw.filled_trigon(self.screen, p1[0], p1[1], p2[0], p2[1], p3[0], p3[1], enemy['color'])

    def _render_projectiles(self):
        for p in self.projectiles:
            pos = p['pos'].astype(int)
            pygame.draw.rect(self.screen, p['color'], (pos[0]-2, pos[1]-p['size'], 4, p['size']*2))
        for p in self.enemy_projectiles:
            pos = p['pos'].astype(int)
            pygame.draw.circle(self.screen, p['color'], pos, p['size'])
    
    def _render_orbs(self):
        for orb in self.energy_orbs:
            pos = orb['pos'].astype(int)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 8, self.COLOR_ENERGY)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 8, self.COLOR_ENERGY)

    def _render_particles(self):
        for p in self.particles:
            pos = p['pos'].astype(int)
            life_ratio = p['life'] / p['max_life']
            size = int(life_ratio * 4)
            if size > 0:
                color = (int(p['color'][0]*life_ratio), int(p['color'][1]*life_ratio), int(p['color'][2]*life_ratio))
                pygame.draw.circle(self.screen, color, pos, size)

    def _render_ui(self):
        # Health and Energy Bars
        bar_width, bar_height = 150, 15
        # Health
        health_pct = max(0, self.player_health / self.player_max_health)
        pygame.draw.rect(self.screen, self.COLOR_UI_BAR, (10, self.HEIGHT - 25, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_UI_HEALTH, (10, self.HEIGHT - 25, int(bar_width * health_pct), bar_height))
        # Energy
        energy_pct = max(0, self.player_energy / self.player_max_energy)
        pygame.draw.rect(self.screen, self.COLOR_UI_BAR, (10, self.HEIGHT - 45, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_UI_ENERGY, (10, self.HEIGHT - 45, int(bar_width * energy_pct), bar_height))

        # Special Ability Meter
        special_pct = max(0, self.player_special_cooldown / 300)
        pygame.draw.rect(self.screen, self.COLOR_UI_BAR, (170, self.HEIGHT - 25, bar_width / 2, bar_height))
        if special_pct <= 0:
            pygame.draw.rect(self.screen, self.COLOR_UI_SPECIAL, (170, self.HEIGHT - 25, bar_width / 2, bar_height))

        # Text Info
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        galaxy_text = self.font_small.render(f"GALAXY: {self.galaxy_number}", True, self.COLOR_TEXT)
        self.screen.blit(galaxy_text, (self.WIDTH - galaxy_text.get_width() - 10, 10))

        if self.game_over and self.player_health <= 0:
            over_text = self.font_large.render("GAME OVER", True, self.COLOR_ENEMY_WEAK)
            self.screen.blit(over_text, (self.WIDTH/2 - over_text.get_width()/2, self.HEIGHT/2 - over_text.get_height()/2))
            
    def _get_info(self):
        self.score = int(self.total_energy_collected / 10) # Simple score metric
        return {
            "score": self.score,
            "steps": self.steps,
            "galaxy": self.galaxy_number,
            "total_energy": self.total_energy_collected,
            "health": self.player_health,
            "energy": self.player_energy
        }

    def _generate_stars(self, n):
        self.stars = []
        for _ in range(n):
            depth = random.uniform(0.1, 0.8)
            self.stars.append({
                'pos': np.array([random.randrange(self.WIDTH), random.randrange(self.HEIGHT)]),
                'size': max(1, int(depth * 2)),
                'depth': depth,
                'color': (int(depth * 100 + 50), int(depth * 100 + 50), int(depth * 150 + 100))
            })

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # This method is for self-checking during development and can be removed.
        print("Running internal validation...")
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
    # Example usage:
    # This block is for human play and debugging, it will not be run by the test suite.
    # Set the video driver to a real one to see the game window.
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    env = GameEnv()
    obs, info = env.reset()
    
    running = True
    # To control the game manually:
    # 0: none, 1: up, 2: down, 3: left, 4: right
    # space: fire, shift: special
    movement = 0
    space_held = 0
    shift_held = 0

    # Create a display for human playing
    pygame.display.set_caption("Cosmic Raider")
    human_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            # Key presses for human control
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE: space_held = 1
                if event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT: shift_held = 1
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_SPACE: space_held = 0
                if event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT: shift_held = 0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        else: movement = 0

        action = [movement, space_held, shift_held]
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation to the human-visible screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        human_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(env.FPS)

        if terminated or truncated:
            print(f"Game Over. Final Info: {info}")
            obs, info = env.reset()
            
    env.close()