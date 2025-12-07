
# Generated: 2025-08-28T06:21:43.401797
# Source Brief: brief_02903.md
# Brief Index: 2903

        
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
        "Controls: Arrow keys to move. Hold Shift to dash. Press Space to attack."
    )

    game_description = (
        "Explore a procedurally generated isometric dungeon. Battle enemies and bosses to reach the final chamber."
    )

    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    COLOR_BG = (15, 15, 25)
    COLOR_WALL = (40, 40, 60)
    COLOR_FLOOR = (25, 25, 40)
    COLOR_PLAYER = (0, 200, 255)
    COLOR_PLAYER_SHADOW = (0, 100, 128)
    COLOR_ENEMY_A = (255, 50, 50)
    COLOR_ENEMY_B = (255, 150, 50)
    COLOR_BOSS = (200, 0, 255)
    COLOR_SHADOW = (10, 10, 15)
    COLOR_TEXT = (220, 220, 240)
    COLOR_HEALTH_GREEN = (50, 200, 50)
    COLOR_HEALTH_RED = (200, 50, 50)
    COLOR_HEALTH_BG = (70, 70, 70)

    TILE_WIDTH = 48
    TILE_HEIGHT = 24
    TILE_WIDTH_HALF = TILE_WIDTH // 2
    TILE_HEIGHT_HALF = TILE_HEIGHT // 2

    MAP_SIZE = 20
    MAX_STEPS = 5000
    MAX_LEVEL = 3

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_damage = pygame.font.SysFont("monospace", 14, bold=True)

        # Game state variables are initialized in reset()
        self.player = {}
        self.enemies = []
        self.particles = []
        self.floating_texts = []
        self.dungeon_layout = np.zeros((self.MAP_SIZE, self.MAP_SIZE), dtype=int)
        self.steps = 0
        self.score = 0
        self.level = 1
        self.game_over = False
        self.camera_offset = [0, 0]
        self.screen_shake = 0

        # Initialize state
        self.reset()
        
        # self.validate_implementation() # Uncomment for self-check

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.level = 1
        self.game_over = False

        self._generate_level()
        
        player_start_pos = self._find_valid_spawn()
        self.player = {
            "x": player_start_pos[0], "y": player_start_pos[1],
            "render_x": player_start_pos[0], "render_y": player_start_pos[1],
            "health": 100, "max_health": 100,
            "attack_cooldown": 0, "dash_cooldown": 0, "invuln_timer": 0,
            "hit_timer": 0, "last_dir": (0, -1)
        }

        self.particles = []
        self.floating_texts = []
        self.screen_shake = 0

        self._spawn_enemies()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0

        # --- Update Game Logic ---
        self.steps += 1
        
        # Cooldowns and timers
        self.player['attack_cooldown'] = max(0, self.player['attack_cooldown'] - 1)
        self.player['dash_cooldown'] = max(0, self.player['dash_cooldown'] - 1)
        self.player['invuln_timer'] = max(0, self.player['invuln_timer'] - 1)
        self.player['hit_timer'] = max(0, self.player['hit_timer'] - 1)

        # Handle player input
        reward += self._handle_player_input(movement, space_held, shift_held)

        # Update enemies
        reward += self._update_enemies()
        
        # Update particles and floating texts
        self._update_particles()
        self._update_floating_texts()

        # Update smooth rendering positions
        self.player['render_x'] += (self.player['x'] - self.player['render_x']) * 0.25
        self.player['render_y'] += (self.player['y'] - self.player['render_y']) * 0.25
        for enemy in self.enemies:
            enemy['render_x'] += (enemy['x'] - enemy['render_x']) * 0.2
            enemy['render_y'] += (enemy['y'] - enemy['render_y']) * 0.2

        # Check for level progression
        if not self.enemies:
            if self.level == self.MAX_LEVEL:
                self.game_over = True
                reward += 100 # Final boss defeated reward
            else:
                self.level += 1
                self._generate_level()
                player_start_pos = self._find_valid_spawn()
                self.player['x'], self.player['y'] = player_start_pos
                self.player['render_x'], self.player['render_y'] = player_start_pos
                self._spawn_enemies()

        # Check termination conditions
        terminated = self._check_termination()
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_player_input(self, movement, space_held, shift_held):
        reward = 0
        dx, dy = 0, 0
        if movement == 1: dy = -1 # Up
        elif movement == 2: dy = 1 # Down
        elif movement == 3: dx = -1 # Left
        elif movement == 4: dx = 1 # Right

        if dx != 0 or dy != 0:
            self.player['last_dir'] = (dx, dy)
            new_x, new_y = self.player['x'] + dx, self.player['y'] + dy
            if self.dungeon_layout[int(new_y), int(new_x)] == 1:
                self.player['x'] = new_x
                self.player['y'] = new_y

        if shift_held and self.player['dash_cooldown'] == 0:
            # Dash action
            self.player['dash_cooldown'] = 60 # 2 seconds
            self.player['invuln_timer'] = 15 # 0.5 seconds of invulnerability
            dash_dist = 3
            dash_x, dash_y = self.player['x'] + self.player['last_dir'][0] * dash_dist, self.player['y'] + self.player['last_dir'][1] * dash_dist
            
            # Check for valid dash destination
            if 0 <= dash_x < self.MAP_SIZE and 0 <= dash_y < self.MAP_SIZE and self.dungeon_layout[int(dash_y), int(dash_x)] == 1:
                self.player['x'], self.player['y'] = dash_x, dash_y
            
            # Dash particles
            for _ in range(20):
                self.particles.append(self._create_particle(self.player['render_x'], self.player['render_y'], self.COLOR_PLAYER, 15, 0.5))
            # Sound: Dash sfx

        if space_held and self.player['attack_cooldown'] == 0:
            # Attack action
            self.player['attack_cooldown'] = 15 # 0.5 sec cooldown
            attack_range = 1.5
            attack_damage = 10
            
            # Attack particles
            for _ in range(5):
                p_dir = (self.player['last_dir'][0] + self.np_random.uniform(-0.5, 0.5), self.player['last_dir'][1] + self.np_random.uniform(-0.5, 0.5))
                self.particles.append(self._create_particle(
                    self.player['render_x'] + self.player['last_dir'][0] * 0.5,
                    self.player['render_y'] + self.player['last_dir'][1] * 0.5,
                    (255, 255, 100), 20, 1.0, vel=p_dir
                ))
            # Sound: Sword swing sfx

            for enemy in self.enemies:
                dist_x = enemy['x'] - self.player['x']
                dist_y = enemy['y'] - self.player['y']
                if math.sqrt(dist_x**2 + dist_y**2) < attack_range:
                    enemy['health'] -= attack_damage
                    enemy['hit_timer'] = 10
                    reward += 0.1 # Reward for damaging enemy
                    self.floating_texts.append(self._create_floating_text(str(attack_damage), enemy['render_x'], enemy['render_y'], self.COLOR_TEXT))
                    # Sound: Hit sfx
        return reward

    def _update_enemies(self):
        reward = 0
        enemies_to_remove = []
        for i, enemy in enumerate(self.enemies):
            enemy['hit_timer'] = max(0, enemy['hit_timer'] - 1)
            
            if enemy['health'] <= 0:
                enemies_to_remove.append(i)
                reward += 10 if enemy['is_boss'] else 1
                self.score += 10 if enemy['is_boss'] else 1
                for _ in range(30):
                    self.particles.append(self._create_particle(enemy['render_x'], enemy['render_y'], enemy['color'], 30, 1.0))
                # Sound: Enemy death explosion
                continue

            dist_to_player = math.sqrt((self.player['x'] - enemy['x'])**2 + (self.player['y'] - enemy['y'])**2)
            
            # --- AI Logic ---
            enemy['cooldown'] = max(0, enemy['cooldown'] - 1)
            
            if dist_to_player < enemy['aggro_range']:
                # Move towards player
                if enemy['cooldown'] == 0:
                    enemy['cooldown'] = enemy['move_speed']
                    dx, dy = self.player['x'] - enemy['x'], self.player['y'] - enemy['y']
                    norm = math.sqrt(dx**2 + dy**2)
                    if norm > 0:
                        move_x = round(dx/norm)
                        move_y = round(dy/norm)
                        new_x, new_y = enemy['x'] + move_x, enemy['y'] + move_y
                        if self.dungeon_layout[int(new_y), int(new_x)] == 1:
                            enemy['x'], enemy['y'] = new_x, new_y
                
                # Attack player
                if dist_to_player < enemy['attack_range'] and enemy['attack_cooldown'] == 0:
                    enemy['attack_cooldown'] = enemy['attack_speed']
                    if self.player['invuln_timer'] == 0:
                        damage = self.np_random.integers(enemy['damage'][0], enemy['damage'][1] + 1)
                        self.player['health'] -= damage
                        self.player['hit_timer'] = 10
                        reward -= 0.02
                        self.screen_shake = 5
                        self.floating_texts.append(self._create_floating_text(str(damage), self.player['render_x'], self.player['render_y'], self.COLOR_HEALTH_RED))
                        # Sound: Player hurt sfx

            enemy['attack_cooldown'] = max(0, enemy['attack_cooldown'] - 1)

        # Remove defeated enemies
        for i in sorted(enemies_to_remove, reverse=True):
            del self.enemies[i]
        
        return reward

    def _check_termination(self):
        if self.player['health'] <= 0:
            self.game_over = True
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
        return self.game_over

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "level": self.level, "health": self.player.get('health', 0)}

    # --- Rendering ---

    def _world_to_screen(self, x, y, z=0):
        iso_x = self.camera_offset[0] + (x - y) * self.TILE_WIDTH_HALF
        iso_y = self.camera_offset[1] + (x + y) * self.TILE_HEIGHT_HALF - z
        return int(iso_x), int(iso_y)

    def _render_game(self):
        # Update camera to follow player
        target_cam_x = self.SCREEN_WIDTH / 2 - (self.player['render_x'] - self.player['render_y']) * self.TILE_WIDTH_HALF
        target_cam_y = self.SCREEN_HEIGHT / 2 - (self.player['render_x'] + self.player['render_y']) * self.TILE_HEIGHT_HALF
        self.camera_offset[0] += (target_cam_x - self.camera_offset[0]) * 0.1
        self.camera_offset[1] += (target_cam_y - self.camera_offset[1]) * 0.1

        # Apply screen shake
        if self.screen_shake > 0:
            shake_offset_x = self.np_random.uniform(-self.screen_shake, self.screen_shake)
            shake_offset_y = self.np_random.uniform(-self.screen_shake, self.screen_shake)
            self.camera_offset[0] += shake_offset_x
            self.camera_offset[1] += shake_offset_y
            self.screen_shake = max(0, self.screen_shake - 1)

        # --- Collect all renderable objects ---
        renderables = []

        # Dungeon tiles
        for y in range(self.MAP_SIZE):
            for x in range(self.MAP_SIZE):
                if self.dungeon_layout[y, x] > 0:
                    renderables.append({'type': 'tile', 'x': x, 'y': y, 'value': self.dungeon_layout[y, x]})

        # Shadows and characters
        all_chars = self.enemies + [self.player]
        for char in all_chars:
            is_player = 'max_health' in char # Simple check for player
            renderables.append({'type': 'shadow', 'x': char['render_x'], 'y': char['render_y'], 'is_boss': char.get('is_boss', False)})
            renderables.append({'type': 'char', 'x': char['render_x'], 'y': char['render_y'], 'char_data': char, 'is_player': is_player})

        # Sort by y-index for correct isometric rendering
        renderables.sort(key=lambda r: r['x'] + r['y'])

        # --- Draw everything in order ---
        for item in renderables:
            if item['type'] == 'tile':
                self._draw_tile(item['x'], item['y'], self.COLOR_FLOOR if item['value'] == 1 else self.COLOR_WALL)
            elif item['type'] == 'shadow':
                self._draw_shadow(item['x'], item['y'], item['is_boss'])
            elif item['type'] == 'char':
                self._draw_character(item['char_data'], item['is_player'])

        # Draw particles and floating texts on top
        for p in self.particles:
            self._draw_particle(p)
        for ft in self.floating_texts:
            self._draw_floating_text(ft)

    def _draw_tile(self, x, y, color):
        sx, sy = self._world_to_screen(x, y)
        points = [
            (sx, sy),
            (sx + self.TILE_WIDTH_HALF, sy + self.TILE_HEIGHT_HALF),
            (sx, sy + self.TILE_HEIGHT),
            (sx - self.TILE_WIDTH_HALF, sy + self.TILE_HEIGHT_HALF)
        ]
        pygame.gfxdraw.filled_polygon(self.screen, points, color)
        pygame.gfxdraw.aapolygon(self.screen, points, (color[0]*1.2, color[1]*1.2, color[2]*1.2))

    def _draw_shadow(self, x, y, is_boss=False):
        sx, sy = self._world_to_screen(x, y)
        radius = self.TILE_WIDTH_HALF * 0.8 if is_boss else self.TILE_WIDTH_HALF * 0.5
        pygame.gfxdraw.filled_ellipse(self.screen, sx, int(sy + self.TILE_HEIGHT), int(radius), int(radius * 0.5), self.COLOR_SHADOW)

    def _draw_character(self, char, is_player):
        height_bob = math.sin(self.steps * 0.1 + char['x']) * 2 if is_player else math.sin(self.steps * 0.05 + char['x']) * 1
        z = self.TILE_HEIGHT * 0.7 + height_bob
        sx, sy = self._world_to_screen(char['render_x'], char['render_y'], z)
        
        size = 12 if is_player else (16 if char.get('is_boss', False) else 8)
        color = self.COLOR_PLAYER if is_player else char['color']

        if char['hit_timer'] > 0:
            color = (255, 255, 255)
        elif is_player and self.player['invuln_timer'] > 0:
            alpha = 150 + math.sin(self.steps * 0.8) * 105
            color = (min(255, color[0] + 50), min(255, color[1] + 50), 255)
            # Cannot do alpha with gfxdraw, so we just brighten
        
        # Simple cube shape
        points_top = [
            self._world_to_screen(char['render_x']-0.3, char['render_y']-0.3, z+size),
            self._world_to_screen(char['render_x']+0.3, char['render_y']-0.3, z+size),
            self._world_to_screen(char['render_x']+0.3, char['render_y']+0.3, z+size),
            self._world_to_screen(char['render_x']-0.3, char['render_y']+0.3, z+size)
        ]
        pygame.gfxdraw.filled_polygon(self.screen, points_top, color)

        # Health bar
        bar_width = 40
        bar_height = 5
        health_pct = char['health'] / char['max_health']
        bar_x = sx - bar_width // 2
        bar_y = sy - size * 1.5
        
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BG, (bar_x, bar_y, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_GREEN if is_player else self.COLOR_HEALTH_RED, (bar_x, bar_y, int(bar_width * health_pct), bar_height))

    def _render_ui(self):
        # Player Health Bar
        health_pct = self.player['health'] / self.player['max_health']
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BG, (10, 10, 200, 20))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_GREEN, (10, 10, int(200 * health_pct), 20))
        health_text = self.font_small.render(f"HP: {max(0, int(self.player['health']))}/{self.player['max_health']}", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (15, 12))

        # Level and Score
        level_text = self.font_large.render(f"Level: {self.level}", True, self.COLOR_TEXT)
        self.screen.blit(level_text, (self.SCREEN_WIDTH - level_text.get_width() - 10, 10))
        score_text = self.font_small.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH - score_text.get_width() - 10, 40))

    # --- Particle System ---
    def _create_particle(self, x, y, color, life, speed, vel=None):
        if vel is None:
            angle = self.np_random.uniform(0, 2 * math.pi)
            vel = (math.cos(angle) * speed, math.sin(angle) * speed)
        return {"x": x, "y": y, "vel_x": vel[0], "vel_y": vel[1], "life": life, "max_life": life, "color": color}

    def _update_particles(self):
        for p in self.particles:
            p['x'] += p['vel_x'] * 0.1
            p['y'] += p['vel_y'] * 0.1
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _draw_particle(self, p):
        sx, sy = self._world_to_screen(p['x'], p['y'], z=self.TILE_HEIGHT*0.5)
        life_pct = p['life'] / p['max_life']
        size = int(3 * life_pct)
        if size > 0:
            color = (p['color'][0], p['color'][1], p['color'][2])
            pygame.draw.circle(self.screen, color, (sx, sy), size)
            
    # --- Floating Text System ---
    def _create_floating_text(self, text, x, y, color):
        return {"text": text, "x": x, "y": y, "life": 45, "color": color}

    def _update_floating_texts(self):
        for ft in self.floating_texts:
            ft['y'] -= 0.05
            ft['life'] -= 1
        self.floating_texts = [ft for ft in self.floating_texts if ft['life'] > 0]

    def _draw_floating_text(self, ft):
        z = self.TILE_HEIGHT * 1.5
        sx, sy = self._world_to_screen(ft['x'], ft['y'], z)
        alpha = int(255 * (ft['life'] / 45))
        text_surf = self.font_damage.render(ft['text'], True, ft['color'])
        text_surf.set_alpha(alpha)
        self.screen.blit(text_surf, (sx - text_surf.get_width()//2, sy - text_surf.get_height()//2))


    # --- Level Generation ---
    def _generate_level(self):
        self.dungeon_layout.fill(0) # 0 = wall
        size = self.MAP_SIZE
        start_x, start_y = size // 2, size // 2
        
        # Random walk to carve out floor
        px, py = start_x, start_y
        num_steps = (size * size) * 2
        for _ in range(num_steps):
            self.dungeon_layout[py, px] = 1 # 1 = floor
            direction = self.np_random.integers(0, 4)
            if direction == 0: px = max(1, px - 1)
            elif direction == 1: px = min(size - 2, px + 1)
            elif direction == 2: py = max(1, py - 1)
            else: py = min(size - 2, py + 1)
            
        # Ensure start is open
        self.dungeon_layout[start_y, start_x] = 1

    def _find_valid_spawn(self):
        while True:
            x = self.np_random.integers(1, self.MAP_SIZE - 1)
            y = self.np_random.integers(1, self.MAP_SIZE - 1)
            if self.dungeon_layout[y, x] == 1:
                return x, y

    def _spawn_enemies(self):
        self.enemies = []
        num_enemies = 3 + self.level * 2

        for _ in range(num_enemies):
            pos = self._find_valid_spawn()
            base_health = 20 + (self.level - 1) * 10
            base_speed = 0.2 + (self.level - 1) * 0.05
            
            self.enemies.append({
                "x": pos[0], "y": pos[1],
                "render_x": pos[0], "render_y": pos[1],
                "health": base_health, "max_health": base_health,
                "color": self.COLOR_ENEMY_A,
                "cooldown": 0, "move_speed": self.np_random.integers(20, 30),
                "attack_cooldown": 0, "attack_speed": 60, "attack_range": 1.5,
                "damage": (5, 10), "aggro_range": 8, "hit_timer": 0, "is_boss": False
            })

        if self.level == self.MAX_LEVEL:
            # Spawn a boss instead of regular enemies
            self.enemies = []
            pos = self._find_valid_spawn()
            boss_health = 200
            self.enemies.append({
                "x": pos[0], "y": pos[1],
                "render_x": pos[0], "render_y": pos[1],
                "health": boss_health, "max_health": boss_health,
                "color": self.COLOR_BOSS,
                "cooldown": 0, "move_speed": 40,
                "attack_cooldown": 0, "attack_speed": 90, "attack_range": 2.0,
                "damage": (15, 25), "aggro_range": 15, "hit_timer": 0, "is_boss": True
            })

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        print("Running implementation validation...")
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
    # This block allows you to play the game manually
    env = GameEnv()
    obs, info = env.reset()
    
    # Setup Pygame window for human play
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Isometric Dungeon Crawler")
    clock = pygame.time.Clock()

    terminated = False
    total_reward = 0
    
    # Action state
    movement = 0 # 0: none, 1: up, 2: down, 3: left, 4: right
    space_held = 0 # 0: released, 1: held
    shift_held = 0 # 0: released, 1: held

    print(GameEnv.user_guide)

    while not terminated:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

        keys = pygame.key.get_pressed()
        
        # Update actions based on key presses
        movement = 0
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        clock.tick(30) # Run at 30 FPS

        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")

    env.close()