import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to move the placement cursor. "
        "Press Shift to cycle through tower types. "
        "Press Space to build the selected tower."
    )

    game_description = (
        "Defend your base from waves of geometric enemies by placing musical towers. "
        "Survive 10 waves to win."
    )

    auto_advance = True

    # --- Constants ---
    # Colors
    COLOR_BG = (15, 19, 25)
    COLOR_PATH = (40, 45, 55)
    COLOR_BASE = (60, 180, 75)
    COLOR_BASE_GLOW = (60, 180, 75, 50)
    COLOR_ENEMY = (220, 50, 50)
    COLOR_CURSOR = (255, 255, 255)
    COLOR_TEXT = (230, 230, 230)
    COLOR_TEXT_WARN = (255, 100, 100)
    
    # Screen Dimensions
    WIDTH, HEIGHT = 640, 400

    # Game Parameters
    MAX_STEPS = 30 * 60 * 2 # 2 minutes at 30fps
    BASE_START_HEALTH = 100
    STARTING_RESOURCES = 200
    NUM_WAVES = 10

    # Tower Specifications
    TOWER_SPECS = [
        {'name': 'Pulse Cannon', 'cost': 50, 'range': 80, 'damage': 12, 'fire_rate': 30, 'color': (0, 150, 255), 'proj_speed': 5},
        {'name': 'Laser Gatling', 'cost': 80, 'range': 100, 'damage': 5, 'fire_rate': 10, 'color': (255, 0, 255), 'proj_speed': 8},
        {'name': 'Heavy Mortar', 'cost': 125, 'range': 120, 'damage': 40, 'fire_rate': 90, 'color': (255, 150, 0), 'proj_speed': 3},
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        
        self.font_ui = pygame.font.SysFont("Consolas", 16)
        self.font_title = pygame.font.SysFont("Consolas", 48, bold=True)
        self.font_subtitle = pygame.font.SysFont("Consolas", 24)

        self._define_path()
        # self.reset() is called by the wrapper, no need to call it here.
        
    def _define_path(self):
        self.path_points = [
            pygame.math.Vector2(-20, self.HEIGHT // 2),
            pygame.math.Vector2(100, self.HEIGHT // 2),
            pygame.math.Vector2(150, 100),
            pygame.math.Vector2(self.WIDTH - 150, 100),
            pygame.math.Vector2(self.WIDTH - 100, self.HEIGHT // 2),
            pygame.math.Vector2(self.WIDTH - 250, self.HEIGHT - 100),
            pygame.math.Vector2(250, self.HEIGHT - 100),
            pygame.math.Vector2(self.WIDTH // 2, self.HEIGHT - 20),
        ]
        self.base_pos = pygame.math.Vector2(self.WIDTH // 2, self.HEIGHT - 50)


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        
        self.base_health = self.BASE_START_HEALTH
        self.resources = self.STARTING_RESOURCES
        
        self.current_wave = -1
        self.wave_cooldown = 90 # 3 seconds at 30fps
        self.enemies_in_wave = 0
        self.enemies_to_spawn = []

        self.towers = []
        self.enemies = []
        self.projectiles = []
        self.particles = []

        self.cursor_pos = pygame.math.Vector2(self.WIDTH // 2, self.HEIGHT // 2)
        self.selected_tower_type = 0
        
        self.last_space_press = False
        self.last_shift_press = False
        self.afford_flash_timer = 0
        
        self._start_next_wave()

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = -0.01  # Small penalty for time passing
        
        self._handle_input(action)

        reward += self._update_towers()
        reward += self._update_projectiles()
        reward += self._update_enemies()
        self._update_particles()
        self._update_waves()

        self.steps += 1
        self.afford_flash_timer = max(0, self.afford_flash_timer - 1)
        
        terminated = self._check_termination()
        if terminated:
            if self.win:
                reward += 100
            else: # Loss
                reward -= 100

        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # --- Cursor Movement ---
        cursor_speed = 5
        if movement == 1: self.cursor_pos.y -= cursor_speed
        elif movement == 2: self.cursor_pos.y += cursor_speed
        elif movement == 3: self.cursor_pos.x -= cursor_speed
        elif movement == 4: self.cursor_pos.x += cursor_speed
        self.cursor_pos.x = np.clip(self.cursor_pos.x, 0, self.WIDTH)
        self.cursor_pos.y = np.clip(self.cursor_pos.y, 0, self.HEIGHT)

        # --- Cycle Tower ---
        if shift_held and not self.last_shift_press:
            self.selected_tower_type = (self.selected_tower_type + 1) % len(self.TOWER_SPECS)
        self.last_shift_press = shift_held

        # --- Place Tower ---
        if space_held and not self.last_space_press:
            spec = self.TOWER_SPECS[self.selected_tower_type]
            if self.resources >= spec['cost'] and self._is_valid_placement(self.cursor_pos):
                self.resources -= spec['cost']
                self.towers.append({
                    'pos': pygame.math.Vector2(self.cursor_pos),
                    'type': self.selected_tower_type,
                    'cooldown': 0,
                    'fire_pulse': 0,
                })
                # sfx: place_tower.wav
            else:
                self.afford_flash_timer = 15 # Flash for 0.5s
                # sfx: error.wav

        self.last_space_press = space_held

    def _is_valid_placement(self, pos):
        # Cannot build on path
        for i in range(len(self.path_points) - 1):
            p1 = self.path_points[i]
            p2 = self.path_points[i+1]
            # Simple bounding box check around path segment
            if pygame.Rect(min(p1.x, p2.x) - 20, min(p1.y, p2.y) - 20, abs(p1.x - p2.x) + 40, abs(p1.y - p2.y) + 40).collidepoint(pos):
                return False
        # Cannot build on other towers
        for tower in self.towers:
            if pos.distance_to(tower['pos']) < 20:
                return False
        # Cannot build near base
        if pos.distance_to(self.base_pos) < 40:
            return False
        return True

    def _update_towers(self):
        for tower in self.towers:
            spec = self.TOWER_SPECS[tower['type']]
            tower['cooldown'] = max(0, tower['cooldown'] - 1)
            tower['fire_pulse'] = max(0, tower['fire_pulse'] - 0.1)

            if tower['cooldown'] == 0:
                target = None
                min_dist = spec['range']
                for enemy in self.enemies:
                    dist = tower['pos'].distance_to(enemy['pos'])
                    if dist < min_dist:
                        min_dist = dist
                        target = enemy
                
                if target:
                    tower['cooldown'] = spec['fire_rate']
                    tower['fire_pulse'] = 1.0
                    self.projectiles.append({
                        'pos': pygame.math.Vector2(tower['pos']),
                        'target': target,
                        'type': tower['type'],
                        'color': spec['color'],
                        'speed': spec['proj_speed'],
                        'damage': spec['damage']
                    })
                    # sfx: tower_fire_{spec['name']}.wav
        return 0 # No immediate reward for firing

    def _update_projectiles(self):
        reward = 0
        for proj in self.projectiles[:]:
            if proj['target'] not in self.enemies: # Target already destroyed
                self.projectiles.remove(proj)
                continue
            
            direction = (proj['target']['pos'] - proj['pos']).normalize()
            proj['pos'] += direction * proj['speed']
            
            if proj['pos'].distance_to(proj['target']['pos']) < 5:
                proj['target']['health'] -= proj['damage']
                reward += 0.1 # Reward for hitting
                self._create_particles(proj['pos'], proj['color'], 5, 2)
                self.projectiles.remove(proj)
                # sfx: impact.wav
        return reward

    def _update_enemies(self):
        reward = 0
        for enemy in self.enemies[:]:
            if enemy['health'] <= 0:
                self.enemies.remove(enemy)
                self.resources += 10 # Gain resources for kill
                reward += 1 # Reward for kill
                self._create_particles(enemy['pos'], self.COLOR_ENEMY, 15, 4)
                # sfx: enemy_destroy.wav
                continue

            path_idx = enemy['path_index']
            target_pos = self.path_points[path_idx] if path_idx < len(self.path_points) else self.base_pos
            
            dist_to_target = enemy['pos'].distance_to(target_pos)
            
            if dist_to_target < enemy['speed']:
                if path_idx < len(self.path_points):
                    enemy['path_index'] += 1
                else: # Reached the base
                    self.base_health -= 10
                    self.base_health = max(0, self.base_health)
                    reward -= 10 # Penalty for base damage
                    self.enemies.remove(enemy)
                    self._create_particles(self.base_pos, self.COLOR_BASE, 30, 5)
                    # sfx: base_damage.wav
                    continue
            
            if (target_pos - enemy['pos']).length() > 0:
                direction = (target_pos - enemy['pos']).normalize()
                enemy['pos'] += direction * enemy['speed']
        return reward

    def _update_waves(self):
        if len(self.enemies) == 0 and len(self.enemies_to_spawn) == 0:
            if self.current_wave >= self.NUM_WAVES - 1:
                self.win = True
                self.game_over = True
                return

            self.wave_cooldown -= 1
            if self.wave_cooldown <= 0:
                self._start_next_wave()
        
        if self.enemies_to_spawn:
            enemy_to_spawn = self.enemies_to_spawn[0]
            enemy_to_spawn['delay'] -= 1
            if enemy_to_spawn['delay'] <= 0:
                self.enemies.append(enemy_to_spawn['data'])
                self.enemies_to_spawn.pop(0)

    def _start_next_wave(self):
        self.current_wave += 1
        self.wave_cooldown = 150 # 5 seconds
        
        wave_num = self.current_wave + 1
        num_enemies = 5 + wave_num * 2
        enemy_health = 20 * (1.05 ** self.current_wave)
        enemy_speed = 1.0 + self.current_wave * 0.05
        spawn_delay = max(5, 30 - wave_num * 2)

        self.enemies_in_wave = num_enemies
        self.enemies_to_spawn = []
        for i in range(num_enemies):
            self.enemies_to_spawn.append({
                'delay': i * spawn_delay,
                'data': {
                    'pos': pygame.math.Vector2(self.path_points[0]),
                    'health': enemy_health,
                    'max_health': enemy_health,
                    'speed': enemy_speed,
                    'path_index': 1,
                }
            })

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['lifespan'] -= 1
            if p['lifespan'] <= 0:
                self.particles.remove(p)

    def _check_termination(self):
        if self.game_over:
            return True
        if self.base_health <= 0:
            self.game_over = True
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

    def _render_game(self):
        self._render_path()
        self._render_base()
        self._render_towers()
        self._render_enemies()
        self._render_projectiles()
        self._render_particles()
        self._render_cursor()

    def _render_path(self):
        for i in range(len(self.path_points) - 1):
            pygame.draw.line(self.screen, self.COLOR_PATH, self.path_points[i], self.path_points[i+1], 30)
        pygame.draw.line(self.screen, self.COLOR_PATH, self.path_points[-1], self.base_pos, 30)
        for p in self.path_points:
            pygame.draw.circle(self.screen, self.COLOR_PATH, (int(p.x), int(p.y)), 15)

    def _render_base(self):
        pos = (int(self.base_pos.x), int(self.base_pos.y))
        # Glow
        for i in range(10, 0, -1):
            alpha = int(50 * (i / 10))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 15 + i, (*self.COLOR_BASE, alpha))
        # Core
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 15, self.COLOR_BASE)
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 15, self.COLOR_BASE)

    def _render_towers(self):
        for tower in self.towers:
            spec = self.TOWER_SPECS[tower['type']]
            pos = (int(tower['pos'].x), int(tower['pos'].y))
            # Firing pulse
            if tower['fire_pulse'] > 0:
                pulse_radius = int(spec['range'] * (1 - tower['fire_pulse']))
                pulse_alpha = int(100 * tower['fire_pulse'])
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], pulse_radius, (*spec['color'], pulse_alpha))
            # Tower body
            pygame.draw.circle(self.screen, spec['color'], pos, 8)
            pygame.draw.circle(self.screen, self.COLOR_BG, pos, 5)
            pygame.draw.circle(self.screen, spec['color'], pos, 2)

    def _render_enemies(self):
        for enemy in self.enemies:
            pos = (int(enemy['pos'].x), int(enemy['pos'].y))
            pygame.draw.circle(self.screen, self.COLOR_ENEMY, pos, 7)
            # Health bar
            health_ratio = enemy['health'] / enemy['max_health']
            bar_width = 14
            bar_height = 3
            bar_pos = (pos[0] - bar_width // 2, pos[1] - 12)
            pygame.draw.rect(self.screen, (80,0,0), (*bar_pos, bar_width, bar_height))
            pygame.draw.rect(self.screen, self.COLOR_BASE, (*bar_pos, int(bar_width * health_ratio), bar_height))

    def _render_projectiles(self):
        for proj in self.projectiles:
            pos = (int(proj['pos'].x), int(proj['pos'].y))
            pygame.draw.circle(self.screen, proj['color'], pos, 3)
            # Trail
            direction = (proj['target']['pos'] - proj['pos']).normalize() if proj['target']['pos'] != proj['pos'] else pygame.math.Vector2(0, -1)
            for i in range(1, 4):
                trail_pos = proj['pos'] - direction * i * 2
                alpha = 150 - i * 40
                pygame.gfxdraw.filled_circle(self.screen, int(trail_pos.x), int(trail_pos.y), 2, (*proj['color'], alpha))

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p['lifespan'] / p['max_lifespan']))
            color = (*p['color'], alpha)
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'].x), int(p['pos'].y), int(p['radius']), color)
    
    def _render_cursor(self):
        pos = (int(self.cursor_pos.x), int(self.cursor_pos.y))
        spec = self.TOWER_SPECS[self.selected_tower_type]
        
        # Range indicator
        can_afford = self.resources >= spec['cost']
        is_valid = self._is_valid_placement(self.cursor_pos)
        range_color = (0, 255, 0, 50) if can_afford and is_valid else (255, 0, 0, 50)
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], spec['range'], range_color)
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], spec['range'], (*range_color[:3], 150))
        
        # Cursor crosshair
        pygame.draw.line(self.screen, self.COLOR_CURSOR, (pos[0] - 10, pos[1]), (pos[0] + 10, pos[1]), 1)
        pygame.draw.line(self.screen, self.COLOR_CURSOR, (pos[0], pos[1] - 10), (pos[0], pos[1] + 10), 1)

    def _render_ui(self):
        # Top Left: Wave Info
        wave_text = f"Wave: {self.current_wave + 1}/{self.NUM_WAVES}"
        enemies_text = f"Enemies: {len(self.enemies) + len(self.enemies_to_spawn)}"
        self._draw_text(wave_text, (10, 10))
        self._draw_text(enemies_text, (10, 30))

        # Top Right: Resources
        res_color = self.COLOR_TEXT if self.afford_flash_timer == 0 else self.COLOR_TEXT_WARN
        self._draw_text(f"Resources: {self.resources}", (self.WIDTH - 150, 10), color=res_color)

        # Bottom Right: Selected Tower
        spec = self.TOWER_SPECS[self.selected_tower_type]
        self._draw_text(f"Selected: {spec['name']}", (self.WIDTH - 200, self.HEIGHT - 50))
        cost_color = self.COLOR_TEXT if self.resources >= spec['cost'] else self.COLOR_TEXT_WARN
        self._draw_text(f"Cost: {spec['cost']}", (self.WIDTH - 200, self.HEIGHT - 30), color=cost_color)

        # Bottom Center: Base Health
        health_ratio = self.base_health / self.BASE_START_HEALTH
        bar_width = 200
        bar_height = 15
        bar_x = (self.WIDTH - bar_width) // 2
        bar_y = self.HEIGHT - 25
        pygame.draw.rect(self.screen, (80,0,0), (bar_x, bar_y, bar_width, bar_height), border_radius=4)
        pygame.draw.rect(self.screen, self.COLOR_BASE, (bar_x, bar_y, bar_width * health_ratio, bar_height), border_radius=4)
        health_text = f"Base Health: {self.base_health}"
        self._draw_text(health_text, (bar_x + bar_width/2, bar_y + bar_height/2), center=True)

        # Game Over / Win Screen
        if self.game_over:
            s = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            s.fill((0, 0, 0, 180))
            self.screen.blit(s, (0, 0))
            title = "VICTORY" if self.win else "GAME OVER"
            color = self.COLOR_BASE if self.win else self.COLOR_ENEMY
            self._draw_text(title, (self.WIDTH/2, self.HEIGHT/2 - 30), font=self.font_title, color=color, center=True)
            self._draw_text(f"Final Score: {int(self.score)}", (self.WIDTH/2, self.HEIGHT/2 + 20), font=self.font_subtitle, center=True)

    def _draw_text(self, text, pos, font=None, color=None, center=False):
        if font is None: font = self.font_ui
        if color is None: color = self.COLOR_TEXT
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect()
        if center:
            text_rect.center = pos
        else:
            text_rect.topleft = pos
        self.screen.blit(text_surface, text_rect)

    def _create_particles(self, pos, color, count, max_speed):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, max_speed)
            vel = pygame.math.Vector2(math.cos(angle), math.sin(angle)) * speed
            lifespan = random.randint(15, 30)
            self.particles.append({
                'pos': pygame.math.Vector2(pos),
                'vel': vel,
                'radius': random.randint(2, 5),
                'color': color,
                'lifespan': lifespan,
                'max_lifespan': lifespan,
            })

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "base_health": self.base_health,
            "resources": self.resources,
            "wave": self.current_wave + 1,
        }
        
    def close(self):
        pygame.quit()


# Example of how to run the environment
if __name__ == '__main__':
    # Set this to 'human' to interact with the game
    render_mode = "human" # "rgb_array" or "human"
    
    if render_mode == "human":
        # Override the screen to be the display screen
        env = GameEnv()
        env.screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
        pygame.display.set_caption("Tower Defense Game")
    else:
        env = GameEnv()

    obs, info = env.reset()
    done = False
    
    # --- Human Controls ---
    # Map keyboard keys to the MultiDiscrete action space
    key_map = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }

    while not done:
        action = [0, 0, 0] # Default no-op action
        
        if render_mode == "human":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
            
            keys = pygame.key.get_pressed()
            
            # Movement
            for key, move_action in key_map.items():
                if keys[key]:
                    action[0] = move_action
                    break
            
            # Space and Shift
            action[1] = 1 if keys[pygame.K_SPACE] else 0
            action[2] = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
            
        else: # Random agent for rgb_array mode
            action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if render_mode == "human":
            # The _get_observation method already draws everything to env.screen
            # So we just need to flip the display
            pygame.display.flip()
            env.clock.tick(30) # Limit to 30 FPS for human play

    env.close()
    print(f"Game Over. Final Info: {info}")