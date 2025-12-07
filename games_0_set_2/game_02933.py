
# Generated: 2025-08-28T06:26:15.629400
# Source Brief: brief_02933.md
# Brief Index: 2933

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrow keys to move. Hold Space to attack in your last moved direction. Hold Shift to use a health potion."
    )

    game_description = (
        "Explore a procedurally generated isometric dungeon. Battle skeletons, collect gold, and find the exit. Your health is limited, but you can find potions to heal."
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 32, 32
        self.TILE_WIDTH, self.TILE_HEIGHT = 32, 16
        self.TILE_WIDTH_HALF, self.TILE_HEIGHT_HALF = self.TILE_WIDTH // 2, self.TILE_HEIGHT // 2
        self.MAX_STEPS = 2000
        self.PLAYER_MAX_HEALTH = 3
        self.ENEMY_SPAWN_INTERVAL = 40
        self.DIFFICULTY_INTERVAL = 500

        # --- Colors ---
        self.COLOR_BG = (25, 20, 25)
        self.COLOR_FLOOR = (60, 45, 48)
        self.COLOR_WALL = (95, 85, 80)
        self.COLOR_WALL_TOP = (120, 110, 105)
        self.COLOR_PLAYER = (0, 200, 255)
        self.COLOR_PLAYER_SHADOW = (0, 100, 128, 100)
        self.COLOR_ENEMY = (255, 60, 60)
        self.COLOR_ENEMY_SHADOW = (128, 30, 30, 100)
        self.COLOR_GOLD = (255, 223, 0)
        self.COLOR_POTION = (0, 255, 128)
        self.COLOR_EXIT = (180, 0, 255)
        self.COLOR_ATTACK = (255, 255, 255)
        self.COLOR_UI_TEXT = (240, 240, 240)
        self.COLOR_UI_BG = (10, 10, 10, 200)

        # --- Gymnasium Spaces ---
        self.observation_space = Box(low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        try:
            self.font_ui = pygame.font.SysFont("Consolas", 18, bold=True)
            self.font_game_over = pygame.font.SysFont("Consolas", 48, bold=True)
        except pygame.error:
            self.font_ui = pygame.font.Font(None, 24)
            self.font_game_over = pygame.font.Font(None, 60)

        # --- Game State Initialization ---
        self.grid = None
        self.floor_tiles = []
        self.player_pos = None
        self.player_health = 0
        self.player_potions = 0
        self.player_facing_dir = [0, 1]
        self.exit_pos = None
        self.enemies = []
        self.items = {}
        self.particles = []
        self.steps = 0
        self.gold = 0
        self.game_over = False
        self.max_enemies = 1
        self.last_enemy_spawn = 0
        self.attack_cooldown = 0
        self.use_potion_cooldown = 0

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.gold = 0
        self.game_over = False
        self.player_health = self.PLAYER_MAX_HEALTH
        self.player_potions = 1
        self.player_facing_dir = [0, 1]
        self.max_enemies = 1
        self.last_enemy_spawn = 0
        self.attack_cooldown = 0
        self.use_potion_cooldown = 0

        self.enemies = []
        self.items = {"gold": [], "potions": []}
        self.particles = []

        self._generate_dungeon()
        self._place_entities()

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = -0.01  # Cost of living
        self.game_over = self.player_health <= 0

        if self.game_over:
            return self._get_observation(), -100.0, True, False, self._get_info()

        # Unpack action
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # Update cooldowns
        if self.attack_cooldown > 0: self.attack_cooldown -= 1
        if self.use_potion_cooldown > 0: self.use_potion_cooldown -= 1

        # --- Player Actions ---
        reward += self._handle_player_actions(movement, space_held, shift_held)

        # --- Game State Update ---
        reward += self._update_game_state()
        
        # --- Spawning Logic ---
        self._update_spawns()

        # --- Termination Check ---
        terminated = False
        if self.player_pos == self.exit_pos:
            reward += 100.0
            terminated = True
            self.game_over = True
        elif self.player_health <= 0:
            reward += -100.0
            terminated = True
            self.game_over = True
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
        
        self.steps += 1
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_player_actions(self, movement, space_held, shift_held):
        reward = 0
        # --- Movement ---
        move_map = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
        if movement in move_map:
            dx, dy = move_map[movement]
            self.player_facing_dir = [dx, dy]
            next_pos = [self.player_pos[0] + dx, self.player_pos[1] + dy]
            if 0 <= next_pos[0] < self.GRID_WIDTH and 0 <= next_pos[1] < self.GRID_HEIGHT and self.grid[next_pos[1]][next_pos[0]] == 0:
                self.player_pos = next_pos
        
        # --- Attack ---
        if space_held and self.attack_cooldown == 0:
            self.attack_cooldown = 5 # 5 steps cooldown
            attack_pos = [self.player_pos[0] + self.player_facing_dir[0], self.player_pos[1] + self.player_facing_dir[1]]
            # sfx: sword_swing.wav
            self._create_particles(10, attack_pos, self.COLOR_ATTACK, life=5, speed=1.5)
            
            hit_enemy = None
            for enemy in self.enemies:
                if enemy['pos'] == attack_pos:
                    hit_enemy = enemy
                    break
            if hit_enemy:
                # sfx: enemy_hit.wav
                self.enemies.remove(hit_enemy)
                self._create_particles(20, hit_enemy['pos'], self.COLOR_ENEMY, life=15, speed=2)
                reward += 1.0

        # --- Use Potion ---
        if shift_held and self.use_potion_cooldown == 0 and self.player_potions > 0:
            self.use_potion_cooldown = 10 # 10 steps cooldown
            if self.player_health < self.PLAYER_MAX_HEALTH:
                # sfx: potion_drink.wav
                self.player_health += 1
                self.player_potions -= 1
                reward += 5.0
                self._create_particles(20, self.player_pos, self.COLOR_POTION, life=15, speed=1, particle_type='heal')

        return reward

    def _update_game_state(self):
        reward = 0
        # --- Update Particles ---
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1

        # --- Enemy AI & Collision ---
        for enemy in self.enemies:
            if enemy['pos'] == self.player_pos:
                self.player_health -= 1
                # sfx: player_hit.wav
                self._create_particles(10, self.player_pos, self.COLOR_PLAYER, life=10, speed=3)
                # Knockback enemy
                enemy['pos'] = [max(0, min(self.GRID_WIDTH-1, enemy['pos'][0] - self.player_facing_dir[0])),
                                max(0, min(self.GRID_HEIGHT-1, enemy['pos'][1] - self.player_facing_dir[1]))]
            else:
                # Simple "move towards player" AI
                dist_x = self.player_pos[0] - enemy['pos'][0]
                dist_y = self.player_pos[1] - enemy['pos'][1]
                if abs(dist_x) + abs(dist_y) < 8: # Aggro range
                    dx, dy = 0, 0
                    if abs(dist_x) > abs(dist_y):
                        dx = 1 if dist_x > 0 else -1
                    else:
                        dy = 1 if dist_y > 0 else -1
                    
                    next_pos = [enemy['pos'][0] + dx, enemy['pos'][1] + dy]
                    if self.grid[next_pos[1]][next_pos[0]] == 0:
                        enemy['pos'] = next_pos
        
        # --- Item Collection ---
        collected_gold = -1
        for i, gold_pos in enumerate(self.items['gold']):
            if gold_pos == self.player_pos:
                collected_gold = i
                break
        if collected_gold != -1:
            # sfx: coin_pickup.wav
            self.items['gold'].pop(collected_gold)
            self.gold += 1
            reward += 0.1
            self._create_particles(5, self.player_pos, self.COLOR_GOLD, life=10, speed=1.5)

        collected_potion = -1
        for i, potion_pos in enumerate(self.items['potions']):
            if potion_pos == self.player_pos:
                collected_potion = i
                break
        if collected_potion != -1:
            # sfx: item_pickup.wav
            self.items['potions'].pop(collected_potion)
            self.player_potions += 1
            self._create_particles(15, self.player_pos, self.COLOR_POTION, life=15, speed=1)

        return reward

    def _update_spawns(self):
        # Increase max enemies over time
        if self.steps > 0 and self.steps % self.DIFFICULTY_INTERVAL == 0:
            self.max_enemies = min(5, self.max_enemies + 1)

        # Spawn new enemies
        if self.steps - self.last_enemy_spawn > self.ENEMY_SPAWN_INTERVAL:
            if len(self.enemies) < self.max_enemies:
                spawn_pos = self._get_random_floor_tile(min_dist_from_player=10)
                if spawn_pos:
                    self.enemies.append({'pos': spawn_pos})
                    self.last_enemy_spawn = self.steps
                    self._create_particles(15, spawn_pos, self.COLOR_ENEMY, life=20, speed=1)

    def _generate_dungeon(self):
        self.grid = np.ones((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=int)
        self.floor_tiles = []
        
        # Random walker algorithm
        walker_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        num_floors = int(self.GRID_WIDTH * self.GRID_HEIGHT * 0.35)
        
        for _ in range(num_floors):
            if self.grid[walker_pos[1]][walker_pos[0]] == 1:
                self.grid[walker_pos[1]][walker_pos[0]] = 0
                self.floor_tiles.append(walker_pos.copy())
            
            direction = self.np_random.choice(['n', 's', 'e', 'w'])
            if direction == 'n': walker_pos[1] = max(1, walker_pos[1] - 1)
            elif direction == 's': walker_pos[1] = min(self.GRID_HEIGHT - 2, walker_pos[1] + 1)
            elif direction == 'w': walker_pos[0] = max(1, walker_pos[0] - 1)
            elif direction == 'e': walker_pos[0] = min(self.GRID_WIDTH - 2, walker_pos[0] + 1)

    def _get_random_floor_tile(self, min_dist_from_player=0):
        attempts = 0
        while attempts < 100:
            attempts += 1
            tile = random.choice(self.floor_tiles)
            if min_dist_from_player > 0 and self.player_pos:
                dist = abs(tile[0] - self.player_pos[0]) + abs(tile[1] - self.player_pos[1])
                if dist >= min_dist_from_player:
                    return tile
            else:
                return tile
        return None

    def _place_entities(self):
        self.player_pos = self._get_random_floor_tile()
        self.exit_pos = self._get_random_floor_tile(min_dist_from_player=max(self.GRID_WIDTH, self.GRID_HEIGHT) // 2)

        for _ in range(15):
            pos = self._get_random_floor_tile()
            if pos and pos != self.player_pos and pos != self.exit_pos:
                self.items['gold'].append(pos)
        
        for _ in range(3):
            pos = self._get_random_floor_tile(min_dist_from_player=5)
            if pos and pos != self.player_pos and pos != self.exit_pos:
                self.items['potions'].append(pos)
        
        # Initial enemy
        enemy_pos = self._get_random_floor_tile(min_dist_from_player=8)
        if enemy_pos:
            self.enemies.append({'pos': enemy_pos})

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"gold": self.gold, "steps": self.steps, "health": self.player_health}

    def _iso_to_cart(self, iso_pos):
        cart_x = (iso_pos[0] - iso_pos[1]) * self.TILE_WIDTH_HALF
        cart_y = (iso_pos[0] + iso_pos[1]) * self.TILE_HEIGHT_HALF
        # Center the grid on the screen
        cart_x += self.SCREEN_WIDTH // 2 - self.TILE_WIDTH_HALF
        cart_y += self.SCREEN_HEIGHT // 2 - (self.GRID_HEIGHT * self.TILE_HEIGHT_HALF) // 2
        return [int(cart_x), int(cart_y)]

    def _render_game(self):
        # --- Create a render queue ---
        render_queue = []
        
        # Add floor, items, and walls
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                pos = [x, y]
                if self.grid[y][x] == 0: # Floor
                    render_queue.append({'type': 'floor', 'pos': pos, 'sort_key': (y, x, 0)})
                    if pos in self.items['gold']:
                        render_queue.append({'type': 'gold', 'pos': pos, 'sort_key': (y, x, 1)})
                    if pos in self.items['potions']:
                        render_queue.append({'type': 'potion', 'pos': pos, 'sort_key': (y, x, 1)})
                    if pos == self.exit_pos:
                        render_queue.append({'type': 'exit', 'pos': pos, 'sort_key': (y, x, 1)})
                else: # Wall
                    render_queue.append({'type': 'wall', 'pos': pos, 'sort_key': (y, x, 3)})

        # Add dynamic entities
        render_queue.append({'type': 'player', 'pos': self.player_pos, 'sort_key': (self.player_pos[1], self.player_pos[0], 2)})
        for enemy in self.enemies:
            render_queue.append({'type': 'enemy', 'pos': enemy['pos'], 'sort_key': (enemy['pos'][1], enemy['pos'][0], 2)})

        # Sort and render
        render_queue.sort(key=lambda item: item['sort_key'])

        for item in render_queue:
            cart_pos = self._iso_to_cart(item['pos'])
            item_type = item['type']

            if item_type == 'floor':
                self._draw_iso_tile(self.screen, self.COLOR_FLOOR, cart_pos)
            elif item_type == 'wall':
                self._draw_iso_cube(self.screen, self.COLOR_WALL, self.COLOR_WALL_TOP, cart_pos)
            elif item_type == 'exit':
                self._draw_iso_sprite(self.screen, self.COLOR_EXIT, cart_pos, size_mod=1.2, shape='rect')
            elif item_type == 'gold':
                self._draw_iso_sprite(self.screen, self.COLOR_GOLD, cart_pos, size_mod=0.6)
            elif item_type == 'potion':
                self._draw_iso_sprite(self.screen, self.COLOR_POTION, cart_pos, size_mod=0.7, shape='rect')
            elif item_type == 'player':
                self._draw_iso_sprite(self.screen, self.COLOR_PLAYER, cart_pos, shadow_color=self.COLOR_PLAYER_SHADOW)
            elif item_type == 'enemy':
                self._draw_iso_sprite(self.screen, self.COLOR_ENEMY, cart_pos, shadow_color=self.COLOR_ENEMY_SHADOW)

        # Render particles on top of everything
        for p in self.particles:
            if p['type'] == 'heal':
                pygame.draw.circle(self.screen, p['color'], self._iso_to_cart(self.player_pos), p['life'], 1)
            else:
                pygame.draw.circle(self.screen, p['color'], p['pos'], max(1, p['life'] // 3))

    def _draw_iso_tile(self, surface, color, cart_pos):
        points = [
            (cart_pos[0], cart_pos[1] + self.TILE_HEIGHT_HALF),
            (cart_pos[0] + self.TILE_WIDTH_HALF, cart_pos[1]),
            (cart_pos[0] + self.TILE_WIDTH, cart_pos[1] + self.TILE_HEIGHT_HALF),
            (cart_pos[0] + self.TILE_WIDTH_HALF, cart_pos[1] + self.TILE_HEIGHT)
        ]
        pygame.gfxdraw.filled_polygon(surface, points, color)
        pygame.gfxdraw.aapolygon(surface, points, color)

    def _draw_iso_cube(self, surface, side_color, top_color, cart_pos, height=24):
        x, y = cart_pos
        w, h = self.TILE_WIDTH, self.TILE_HEIGHT
        wh, hh = self.TILE_WIDTH_HALF, self.TILE_HEIGHT_HALF

        top_points = [(x, y + hh), (x + wh, y), (x + w, y + hh), (x + wh, y + h)]
        side1_points = [(x, y + hh), (x, y + hh + height), (x + wh, y + h + height), (x + wh, y + h)]
        side2_points = [(x + wh, y + h), (x + wh, y + h + height), (x + w, y + hh + height), (x + w, y + hh)]
        
        pygame.gfxdraw.filled_polygon(surface, side1_points, tuple(c * 0.7 for c in side_color))
        pygame.gfxdraw.filled_polygon(surface, side2_points, tuple(c * 0.85 for c in side_color))
        pygame.gfxdraw.filled_polygon(surface, top_points, top_color)
        pygame.gfxdraw.aapolygon(surface, top_points, top_color)

    def _draw_iso_sprite(self, surface, color, cart_pos, size_mod=1.0, shape='circle', shadow_color=None):
        center_x = cart_pos[0] + self.TILE_WIDTH_HALF
        center_y = cart_pos[1] + self.TILE_HEIGHT_HALF
        
        if shadow_color:
            shadow_radius = int(self.TILE_WIDTH_HALF * 0.8)
            shadow_surf = pygame.Surface((shadow_radius*2, shadow_radius*2), pygame.SRCALPHA)
            pygame.draw.ellipse(shadow_surf, shadow_color, (0,0,shadow_radius*2, shadow_radius))
            surface.blit(shadow_surf, (center_x - shadow_radius, center_y + self.TILE_HEIGHT_HALF - shadow_radius//2))

        radius = int(self.TILE_WIDTH_HALF * 0.4 * size_mod)
        if shape == 'circle':
            pygame.gfxdraw.filled_circle(surface, center_x, center_y, radius, color)
            pygame.gfxdraw.aacircle(surface, center_x, center_y, radius, color)
        elif shape == 'rect':
            size = int(self.TILE_WIDTH * 0.4 * size_mod)
            rect = pygame.Rect(center_x - size//2, center_y - size//2, size, size)
            pygame.draw.rect(surface, color, rect)

    def _render_ui(self):
        ui_surf = pygame.Surface((self.SCREEN_WIDTH, 40), pygame.SRCALPHA)
        ui_surf.fill(self.COLOR_UI_BG)
        
        health_text = self.font_ui.render(f"HEALTH: {'♥' * self.player_health}{'♡' * (self.PLAYER_MAX_HEALTH - self.player_health)}", True, self.COLOR_UI_TEXT)
        ui_surf.blit(health_text, (10, 8))

        potions_text = self.font_ui.render(f"POTIONS: {self.player_potions}", True, self.COLOR_POTION)
        ui_surf.blit(potions_text, (180, 8))

        gold_text = self.font_ui.render(f"GOLD: {self.gold}", True, self.COLOR_GOLD)
        ui_surf.blit(gold_text, (320, 8))

        steps_text = self.font_ui.render(f"STEPS: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_UI_TEXT)
        ui_surf.blit(steps_text, (450, 8))

        self.screen.blit(ui_surf, (0,0))

        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            
            status_text = "YOU WIN!" if self.player_pos == self.exit_pos else "GAME OVER"
            color = (0, 255, 0) if self.player_pos == self.exit_pos else (255, 0, 0)
            
            text_surf = self.font_game_over.render(status_text, True, color)
            text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            overlay.blit(text_surf, text_rect)
            self.screen.blit(overlay, (0,0))

    def _create_particles(self, count, pos, color, life, speed, particle_type='burst'):
        cart_pos = self._iso_to_cart(pos)
        center_pos = [cart_pos[0] + self.TILE_WIDTH_HALF, cart_pos[1] + self.TILE_HEIGHT_HALF]
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            vel = [math.cos(angle) * speed * self.np_random.uniform(0.5, 1.5), 
                   math.sin(angle) * speed * self.np_random.uniform(0.5, 1.5)]
            self.particles.append({'pos': list(center_pos), 'vel': vel, 'life': life, 'color': color, 'type': particle_type})

    def close(self):
        pygame.quit()

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
        assert not trunc
        assert isinstance(info, dict)
        
        # Test state assertions
        assert self.player_health <= self.PLAYER_MAX_HEALTH
        assert self.gold >= 0
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Game loop
    running = True
    while running:
        # Pygame event handling
        action = np.array([0, 0, 0]) # Default no-op
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # Keyboard input
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        elif keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        
        if keys[pygame.K_SPACE]:
            action[1] = 1
        
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            action[2] = 1

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation from the environment
        # Pygame uses (width, height), numpy uses (height, width)
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        
        # Create a display if one doesn't exist
        try:
            display_surf = pygame.display.get_surface()
            if display_surf is None:
                raise AttributeError
            display_surf.blit(surf, (0, 0))
        except (pygame.error, AttributeError):
            display_surf = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
            display_surf.blit(surf, (0, 0))
            
        pygame.display.flip()

        if terminated or truncated:
            print(f"Game Over! Final Gold: {info['gold']}, Steps: {info['steps']}")
            pygame.time.wait(2000) # Pause for 2 seconds
            obs, info = env.reset()

        # Since auto_advance is False, we control the speed
        env.clock.tick(15) # Run at 15 FPS for human playability

    env.close()