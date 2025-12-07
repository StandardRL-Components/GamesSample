
# Generated: 2025-08-27T12:58:50.548854
# Source Brief: brief_00220.md
# Brief Index: 220

        
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
        "Controls: Arrow keys to move the placement cursor. Space to place the selected tower. Shift to cycle tower types."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Defend your base from waves of geometric invaders by strategically placing various types of towers."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 30 * 120 # 2 minutes max
        self.MAX_WAVES = 10
        self.WAVE_PREP_TIME = 5 * self.FPS # 5 seconds

        # Colors
        self.COLOR_BG = (20, 20, 30)
        self.COLOR_PATH = (40, 40, 60)
        self.COLOR_BASE = (0, 200, 100)
        self.COLOR_ENEMY = (255, 50, 50)
        self.COLOR_CURSOR = (255, 255, 0, 150)
        self.COLOR_ZONE = (255, 255, 0, 50)
        self.COLOR_TEXT = (230, 230, 230)
        
        # Tower types: [cost, range, damage, fire_rate (frames), color, name]
        self.TOWER_TYPES = [
            {'cost': 100, 'range': 100, 'damage': 5, 'fire_rate': 10, 'color': (50, 150, 255), 'name': 'Gatling', 'projectile_speed': 10},
            {'cost': 250, 'range': 150, 'damage': 25, 'fire_rate': 60, 'color': (255, 100, 200), 'name': 'Cannon', 'projectile_speed': 7},
            {'cost': 200, 'range': 70, 'damage': 2, 'fire_rate': 20, 'color': (150, 50, 255), 'name': 'AoE Pulse', 'projectile_speed': 0}, # AoE has 0 projectile speed
        ]

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 36)
        
        # Initialize state variables
        self.reset()
        
        # Validate implementation
        # self.validate_implementation() # Commented out for final submission
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.reward_this_step = 0

        self.base_pos = (self.WIDTH - 40, self.HEIGHT / 2)
        self.base_health = 100
        self.resources = 300
        
        self.wave_number = 0
        self.wave_timer = self.WAVE_PREP_TIME
        self.enemies_in_wave = 0
        self.enemies_spawned = 0
        
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []

        self.cursor_pos = np.array([self.WIDTH / 2, self.HEIGHT / 2], dtype=float)
        self.selected_tower_type = 0
        self.prev_space_held = False
        self.prev_shift_held = False

        self._generate_path_and_zones()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        self.reward_this_step = -0.001 # Small penalty for existing to encourage speed

        # Handle player actions
        self._handle_actions(action)
        
        # Update game logic
        self._update_wave_system()
        self._update_towers()
        self._update_projectiles()
        self._update_enemies()
        self._update_particles()
        
        self.steps += 1
        
        # Check for termination
        terminated = self._check_termination()
        
        reward = self.reward_this_step
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_actions(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # Move cursor
        cursor_speed = 5
        if movement == 1: self.cursor_pos[1] -= cursor_speed
        elif movement == 2: self.cursor_pos[1] += cursor_speed
        elif movement == 3: self.cursor_pos[0] -= cursor_speed
        elif movement == 4: self.cursor_pos[0] += cursor_speed
        
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.WIDTH)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.HEIGHT)

        # Place tower (on key press)
        if space_held and not self.prev_space_held:
            self._place_tower()
        self.prev_space_held = space_held
        
        # Cycle tower type (on key press)
        if shift_held and not self.prev_shift_held:
            self.selected_tower_type = (self.selected_tower_type + 1) % len(self.TOWER_TYPES)
        self.prev_shift_held = shift_held

    def _place_tower(self):
        tower_spec = self.TOWER_TYPES[self.selected_tower_type]
        if self.resources >= tower_spec['cost']:
            for zone_pos, zone_radius in self.placement_zones:
                dist = np.linalg.norm(self.cursor_pos - zone_pos)
                if dist < zone_radius:
                    # Check if another tower is already here
                    is_occupied = False
                    for t in self.towers:
                        if np.linalg.norm(np.array(t['pos']) - self.cursor_pos) < 20:
                            is_occupied = True
                            break
                    if not is_occupied:
                        self.resources -= tower_spec['cost']
                        self.towers.append({
                            'pos': self.cursor_pos.copy(),
                            'type': self.selected_tower_type,
                            'cooldown': 0,
                            'target': None
                        })
                        # sfx: tower_placed
                        self._create_particles(self.cursor_pos, 10, tower_spec['color'], 1, 3)
                        break

    def _update_wave_system(self):
        if len(self.enemies) == 0 and self.enemies_spawned == self.enemies_in_wave:
            self.wave_timer -= 1
            if self.wave_timer <= 0:
                if self.wave_number > 0: # Don't reward for completing wave 0
                    self.reward_this_step += 10
                    self.resources += 150 + self.wave_number * 10
                
                self.wave_number += 1
                if self.wave_number > self.MAX_WAVES:
                    return

                self.wave_timer = self.WAVE_PREP_TIME
                self.enemies_in_wave = 5 + self.wave_number * 3
                self.enemies_spawned = 0
                self.spawn_timer = 0
        else:
            self.spawn_timer -= 1
            if self.spawn_timer <= 0 and self.enemies_spawned < self.enemies_in_wave:
                self._spawn_enemy()
                self.enemies_spawned += 1
                self.spawn_timer = max(10, 30 - self.wave_number) # Spawn faster in later waves

    def _spawn_enemy(self):
        health = 20 + (self.wave_number - 1) * 10
        speed = 1.0 + (self.wave_number - 1) * 0.1
        self.enemies.append({
            'pos': self.path[0].copy(),
            'path_index': 0,
            'health': health,
            'max_health': health,
            'speed': speed,
        })
    
    def _update_towers(self):
        for tower in self.towers:
            spec = self.TOWER_TYPES[tower['type']]
            tower['cooldown'] = max(0, tower['cooldown'] - 1)
            
            if tower['cooldown'] == 0:
                # Find target
                target = None
                min_dist = float('inf')
                for enemy in self.enemies:
                    dist = np.linalg.norm(np.array(tower['pos']) - enemy['pos'])
                    if dist < spec['range'] and dist < min_dist:
                        min_dist = dist
                        target = enemy
                
                if target:
                    tower['cooldown'] = spec['fire_rate']
                    # sfx: tower_fire
                    if spec['name'] == 'AoE Pulse':
                        # AoE logic
                        self._create_particles(tower['pos'], 1, spec['color'], spec['range'], 0, is_pulse=True)
                        for enemy in self.enemies:
                            if np.linalg.norm(np.array(tower['pos']) - enemy['pos']) < spec['range']:
                                enemy['health'] -= spec['damage']
                                self.reward_this_step += 0.1
                                self._create_particles(enemy['pos'], 3, self.COLOR_ENEMY, 1, 2)
                    else:
                        # Projectile logic
                        self.projectiles.append({
                            'pos': tower['pos'].copy(),
                            'target_pos': target['pos'].copy(),
                            'speed': spec['projectile_speed'],
                            'damage': spec['damage'],
                            'color': spec['color']
                        })
                        self._create_particles(tower['pos'], 3, spec['color'], 1, 2)


    def _update_projectiles(self):
        for p in self.projectiles[:]:
            direction = self.projectiles[0]['target_pos'] - p['pos']
            dist = np.linalg.norm(direction)
            if dist < p['speed']:
                p['pos'] = self.projectiles[0]['target_pos']
            else:
                p['pos'] += (direction / dist) * p['speed']
            
            # Check for collision with any enemy
            hit = False
            for enemy in self.enemies:
                if np.linalg.norm(p['pos'] - enemy['pos']) < 10:
                    enemy['health'] -= p['damage']
                    self.reward_this_step += 0.1
                    hit = True
                    # sfx: projectile_hit
                    self._create_particles(p['pos'], 5, p['color'], 1, 3)
                    break
            if hit:
                self.projectiles.remove(p)


    def _update_enemies(self):
        for e in self.enemies[:]:
            if e['health'] <= 0:
                self.score += 10
                self.resources += 20
                self.reward_this_step += 1
                # sfx: enemy_destroyed
                self._create_particles(e['pos'], 20, self.COLOR_ENEMY, 2, 4)
                self.enemies.remove(e)
                continue

            path_index = e['path_index']
            if path_index >= len(self.path) - 1:
                self.base_health -= 10
                self.reward_this_step -= 10
                self.enemies.remove(e)
                # sfx: base_damage
                self._create_particles(self.base_pos, 30, self.COLOR_BASE, 3, 5, is_pulse=True)
                continue
            
            target_node = self.path[path_index + 1]
            direction = target_node - e['pos']
            dist = np.linalg.norm(direction)
            
            if dist < e['speed']:
                e['pos'] = target_node.copy()
                e['path_index'] += 1
            else:
                e['pos'] += (direction / dist) * e['speed']

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['lifespan'] -= 1
            if p['lifespan'] <= 0:
                self.particles.remove(p)

    def _check_termination(self):
        if self.base_health <= 0:
            self.game_over = True
            self.reward_this_step = -100
            return True
        if self.wave_number > self.MAX_WAVES and len(self.enemies) == 0:
            self.game_over = True
            self.reward_this_step = 100
            return True
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True
        return False

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
            "wave": self.wave_number,
            "health": self.base_health,
            "resources": self.resources,
        }

    def _generate_path_and_zones(self):
        self.path = []
        y = self.HEIGHT / 2
        for x in range(20, self.WIDTH - 20, 20):
            y_offset = 100 * math.sin(x / 80)
            self.path.append(np.array([float(x), y + y_offset]))
        self.path.append(np.array([self.base_pos[0]-20, self.base_pos[1]]))
        
        self.placement_zones = []
        for i in range(5, len(self.path) - 5, 4):
            p = self.path[i]
            offset_dir = np.array([-(self.path[i+1][1] - self.path[i-1][1]), self.path[i+1][0] - self.path[i-1][0]])
            offset_dir /= np.linalg.norm(offset_dir)
            self.placement_zones.append((p + offset_dir * 60, 25))
            self.placement_zones.append((p - offset_dir * 60, 25))

    def _render_game(self):
        # Draw path
        if len(self.path) > 1:
            pygame.draw.lines(self.screen, self.COLOR_PATH, False, [p.astype(int) for p in self.path], 40)
        
        # Draw placement zones
        for pos, radius in self.placement_zones:
            pygame.gfxdraw.filled_circle(self.screen, int(pos[0]), int(pos[1]), radius, self.COLOR_ZONE)
            pygame.gfxdraw.aacircle(self.screen, int(pos[0]), int(pos[1]), radius, self.COLOR_ZONE)

        # Draw base
        pygame.draw.rect(self.screen, self.COLOR_BASE, (self.base_pos[0] - 15, self.base_pos[1] - 15, 30, 30))

        # Draw towers
        for tower in self.towers:
            spec = self.TOWER_TYPES[tower['type']]
            pos = tower['pos'].astype(int)
            pygame.draw.circle(self.screen, spec['color'], pos, 10)
            pygame.draw.circle(self.screen, self.COLOR_TEXT, pos, 10, 2)
            if tower['cooldown'] > spec['fire_rate'] - 5: # Firing flash
                pygame.draw.circle(self.screen, (255,255,255), pos, 12, 2)

        # Draw projectiles
        for p in self.projectiles:
            pygame.draw.line(self.screen, p['color'], p['pos'].astype(int), (p['pos'] - (p['target_pos'] - p['pos'])*0.1).astype(int), 3)

        # Draw enemies
        for e in self.enemies:
            pos = e['pos'].astype(int)
            pygame.draw.rect(self.screen, self.COLOR_ENEMY, (pos[0]-7, pos[1]-7, 14, 14))
            # Health bar
            health_pct = e['health'] / e['max_health']
            pygame.draw.rect(self.screen, (50,50,50), (pos[0]-8, pos[1]-15, 16, 4))
            pygame.draw.rect(self.screen, self.COLOR_BASE, (pos[0]-8, pos[1]-15, 16 * health_pct, 4))

        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p['lifespan'] / p['max_lifespan']))
            color = p['color'] + (alpha,)
            if p.get('is_pulse', False):
                pygame.gfxdraw.aacircle(self.screen, int(p['pos'][0]), int(p['pos'][1]), int(p['radius']), color)
            else:
                size = max(1, int(p['radius'] * (p['lifespan'] / p['max_lifespan'])))
                pygame.draw.circle(self.screen, p['color'], p['pos'].astype(int), size)

        # Draw cursor
        cursor_surf = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        c_pos = self.cursor_pos.astype(int)
        tower_spec = self.TOWER_TYPES[self.selected_tower_type]
        # Show range
        can_afford = self.resources >= tower_spec['cost']
        range_color = (0, 255, 0, 30) if can_afford else (255, 0, 0, 30)
        pygame.gfxdraw.filled_circle(cursor_surf, c_pos[0], c_pos[1], tower_spec['range'], range_color)
        pygame.gfxdraw.aacircle(cursor_surf, c_pos[0], c_pos[1], tower_spec['range'], (*range_color[:3], 100))
        # Show cursor
        pygame.draw.circle(cursor_surf, self.COLOR_CURSOR, c_pos, 5)
        self.screen.blit(cursor_surf, (0,0))
    
    def _render_ui(self):
        # Wave info
        if len(self.enemies) == 0 and self.enemies_spawned == self.enemies_in_wave and self.wave_number <= self.MAX_WAVES:
            wave_text = f"Wave {self.wave_number+1} starting in {self.wave_timer // self.FPS}"
        else:
            wave_text = f"Wave: {self.wave_number}/{self.MAX_WAVES}"
        self._draw_text(wave_text, (10, 10), self.font_small)

        # Base health
        self._draw_text(f"Base Health: {max(0, self.base_health)}/100", (self.WIDTH - 10, 10), self.font_small, align="right")
        
        # Resources
        self._draw_text(f"Resources: {self.resources}", (10, self.HEIGHT - 30), self.font_large)

        # Score
        self._draw_text(f"Score: {self.score}", (self.WIDTH - 10, self.HEIGHT - 30), self.font_large, align="right")
        
        # Selected tower
        tower_spec = self.TOWER_TYPES[self.selected_tower_type]
        tower_info = f"Selected: {tower_spec['name']} (Cost: {tower_spec['cost']})"
        self._draw_text(tower_info, (self.WIDTH / 2, self.HEIGHT - 20), self.font_small, align="center")

        # Game Over / Win message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            if self.base_health <= 0:
                msg = "GAME OVER"
            else:
                msg = "VICTORY!"
            self._draw_text(msg, (self.WIDTH/2, self.HEIGHT/2 - 20), self.font_large, align="center")
            self._draw_text(f"Final Score: {self.score}", (self.WIDTH/2, self.HEIGHT/2 + 20), self.font_small, align="center")

    def _draw_text(self, text, pos, font, color=None, align="left"):
        if color is None: color = self.COLOR_TEXT
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect()
        if align == "right":
            text_rect.topright = pos
        elif align == "center":
            text_rect.midtop = pos
        else:
            text_rect.topleft = pos
        self.screen.blit(text_surface, text_rect)

    def _create_particles(self, pos, count, color, min_speed, max_speed, lifespan=20, is_pulse=False):
        for _ in range(count):
            if is_pulse:
                self.particles.append({
                    'pos': pos, 'vel': np.zeros(2), 'radius': 1, 'lifespan': lifespan, 
                    'max_lifespan': lifespan, 'color': color, 'is_pulse': True
                })
            else:
                angle = random.uniform(0, 2 * math.pi)
                speed = random.uniform(min_speed, max_speed)
                vel = np.array([math.cos(angle), math.sin(angle)]) * speed
                self.particles.append({
                    'pos': pos.copy(), 'vel': vel, 'radius': random.randint(2, 4), 
                    'lifespan': lifespan, 'max_lifespan': lifespan, 'color': color
                })

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to run the file directly to play the game
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen_width, screen_height = 640, 400
    pygame.display.set_caption("Tower Defense")
    real_screen = pygame.display.set_mode((screen_width, screen_height))
    
    done = False
    clock = pygame.time.Clock()
    
    while not done:
        # Map pygame keys to the action space
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Render the observation to the real screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        real_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        clock.tick(env.FPS)

    pygame.quit()
    print(f"Game Over! Final Info: {info}")