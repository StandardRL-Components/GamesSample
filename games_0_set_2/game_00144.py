
# Generated: 2025-08-27T12:43:44.267297
# Source Brief: brief_00144.md
# Brief Index: 144

        
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
        "Controls: Arrow keys to move. Space to attack in your last moved direction. "
        "The game is turn-based; an action must be taken to advance the state."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "An isometric roguelike. Navigate the dungeon, defeat enemies, "
        "collect health potions, and find the exit to win."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    MAP_WIDTH, MAP_HEIGHT = 20, 15
    TILE_WIDTH, TILE_HEIGHT = 32, 16
    TILE_WIDTH_HALF, TILE_HEIGHT_HALF = TILE_WIDTH // 2, TILE_HEIGHT // 2

    COLOR_BG = (25, 20, 25)
    COLOR_FLOOR = (70, 60, 50)
    COLOR_WALL_TOP = (100, 90, 80)
    COLOR_WALL_SIDE = (80, 70, 60)
    COLOR_PLAYER = (50, 200, 255)
    COLOR_PLAYER_GLOW = (50, 200, 255, 100)
    COLOR_ENEMY = (255, 80, 80)
    COLOR_ENEMY_GLOW = (255, 80, 80, 100)
    COLOR_POTION = (80, 255, 80)
    COLOR_POTION_GLOW = (80, 255, 80, 100)
    COLOR_EXIT = (255, 220, 50)
    COLOR_EXIT_GLOW = (255, 220, 50, 100)
    COLOR_ATTACK = (255, 255, 255)
    COLOR_UI_TEXT = (220, 220, 220)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.font = pygame.font.Font(None, 28)
        self.clock = pygame.time.Clock()

        self.map_data = []
        self.player_pos = [0, 0]
        self.player_health = 0
        self.player_facing = (0, 1)
        self.max_player_health = 10
        self.enemies = []
        self.potions = []
        self.exit_pos = [0, 0]
        self.particles = []
        self.steps = 0
        self.score = 0
        self.game_over = False

        # This ensures the first call to reset() has a seed if provided to __init__
        # but gym standard is to seed in reset() itself.
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.particles = []

        self._generate_map()
        self._place_entities()

        self.player_health = 10
        self.max_player_health = 10 + len(self.potions) * 3
        self.player_facing = (0, 1) # Down

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_pressed, _ = action
        reward = 0
        self.steps += 1

        # --- Player Action Phase ---
        moved = False
        if movement != 0:
            moved = self._handle_player_move(movement)
        elif space_pressed:
            self._handle_player_attack()

        # --- Interaction and Reward Phase ---
        # Potion collection
        for potion in self.potions:
            if potion['active'] and self.player_pos == potion['pos']:
                potion['active'] = False
                self.player_health = min(self.max_player_health, self.player_health + 3)
                reward += 1
                # Sound: Potion collect chime

        # Movement reward
        if moved:
            is_risky = any(
                abs(self.player_pos[0] - e['pos'][0]) + abs(self.player_pos[1] - e['pos'][1]) == 1
                for e in self.enemies if e['health'] > 0
            )
            reward += 0.5 if is_risky else -0.2

        # --- Enemy Phase ---
        player_damaged_this_turn = False
        for enemy in self.enemies:
            if enemy['health'] > 0:
                self._update_enemy_movement(enemy)
                if abs(self.player_pos[0] - enemy['pos'][0]) + abs(self.player_pos[1] - enemy['pos'][1]) == 1:
                    if not player_damaged_this_turn:
                         self.player_health -= 1
                         player_damaged_this_turn = True
                         # Sound: Player hurt grunt

        # Enemy attack reward is implicitly handled by player death penalty

        # Attack damage resolution
        dead_enemies_this_turn = self._resolve_attacks()
        reward += dead_enemies_this_turn * 2

        # --- Termination Check ---
        terminated = False
        if self.player_pos == self.exit_pos:
            reward += 100
            terminated = True
            # Sound: Victory fanfare
        elif self.player_health <= 0:
            reward = -100 # Override other rewards
            terminated = True
            # Sound: Player death sound
        elif self.steps >= 1000:
            terminated = True

        self.game_over = terminated
        self.score += reward

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_player_move(self, movement):
        move_map = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)} # Up, Down, Left, Right
        dx, dy = move_map[movement]
        self.player_facing = (dx, dy)
        
        new_pos = [self.player_pos[0] + dx, self.player_pos[1] + dy]

        if (0 <= new_pos[0] < self.MAP_WIDTH and
            0 <= new_pos[1] < self.MAP_HEIGHT and
            self.map_data[new_pos[1]][new_pos[0]] == 0):
            self.player_pos = new_pos
            assert 0 <= self.player_pos[0] < self.MAP_WIDTH and 0 <= self.player_pos[1] < self.MAP_HEIGHT
            return True
        return False

    def _handle_player_attack(self):
        # Sound: Player attack swoosh
        attack_pos = [self.player_pos[0] + self.player_facing[0], self.player_pos[1] + self.player_facing[1]]
        screen_pos = self._world_to_screen(attack_pos[0], attack_pos[1])
        for _ in range(15):
            self.particles.append({
                'pos': [screen_pos[0], screen_pos[1] - self.TILE_HEIGHT_HALF],
                'vel': [self.np_random.uniform(-2, 2), self.np_random.uniform(-2, 2)],
                'life': 15,
                'color': self.COLOR_ATTACK,
                'radius': self.np_random.integers(1, 4),
                'attack_pos': attack_pos
            })

    def _update_enemy_movement(self, enemy):
        path = enemy['patrol_path']
        current_idx = enemy['patrol_idx']
        next_pos = path[(current_idx + 1) % len(path)]

        if self.map_data[next_pos[1]][next_pos[0]] == 0:
            enemy['pos'] = next_pos
            enemy['patrol_idx'] = (current_idx + 1) % len(path)

    def _resolve_attacks(self):
        dead_enemies = 0
        for particle in self.particles:
            if 'attack_pos' in particle:
                for enemy in self.enemies:
                    if enemy['health'] > 0 and enemy['pos'] == particle['attack_pos']:
                        enemy['health'] -= 1
                        if enemy['health'] <= 0:
                            dead_enemies += 1
                            # Sound: Enemy defeat explosion
                # Ensure attack only hits once
                del particle['attack_pos']
        return dead_enemies

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "health": self.player_health}

    def _world_to_screen(self, x, y):
        screen_x = (self.SCREEN_WIDTH // 2) + (x - y) * self.TILE_WIDTH_HALF
        screen_y = (self.SCREEN_HEIGHT // 4) + (x + y) * self.TILE_HEIGHT_HALF
        return int(screen_x), int(screen_y)

    def _draw_iso_cube(self, surface, x, y, color_top, color_side):
        screen_x, screen_y = self._world_to_screen(x, y)
        points_top = [
            (screen_x, screen_y),
            (screen_x + self.TILE_WIDTH_HALF, screen_y + self.TILE_HEIGHT_HALF),
            (screen_x, screen_y + self.TILE_HEIGHT),
            (screen_x - self.TILE_WIDTH_HALF, screen_y + self.TILE_HEIGHT_HALF)
        ]
        points_left = [
            (screen_x - self.TILE_WIDTH_HALF, screen_y + self.TILE_HEIGHT_HALF),
            (screen_x, screen_y + self.TILE_HEIGHT),
            (screen_x, screen_y + self.TILE_HEIGHT + self.TILE_HEIGHT),
            (screen_x - self.TILE_WIDTH_HALF, screen_y + self.TILE_HEIGHT_HALF + self.TILE_HEIGHT)
        ]
        points_right = [
            (screen_x + self.TILE_WIDTH_HALF, screen_y + self.TILE_HEIGHT_HALF),
            (screen_x, screen_y + self.TILE_HEIGHT),
            (screen_x, screen_y + self.TILE_HEIGHT + self.TILE_HEIGHT),
            (screen_x + self.TILE_WIDTH_HALF, screen_y + self.TILE_HEIGHT_HALF + self.TILE_HEIGHT)
        ]
        pygame.gfxdraw.filled_polygon(surface, points_top, color_top)
        pygame.gfxdraw.aapolygon(surface, points_top, color_top)
        pygame.gfxdraw.filled_polygon(surface, points_left, color_side)
        pygame.gfxdraw.aapolygon(surface, points_left, color_side)
        pygame.gfxdraw.filled_polygon(surface, points_right, color_side)
        pygame.gfxdraw.aapolygon(surface, points_right, color_side)

    def _generate_map(self):
        self.map_data = [[1] * self.MAP_WIDTH for _ in range(self.MAP_HEIGHT)]
        start_x, start_y = self.np_random.integers(1, self.MAP_WIDTH-1), self.np_random.integers(1, self.MAP_HEIGHT-1)
        self.map_data[start_y][start_x] = 0
        
        walkers = [{'x': start_x, 'y': start_y}]
        num_floors = (self.MAP_WIDTH * self.MAP_HEIGHT) // 2
        
        created_floors = 1
        while created_floors < num_floors:
            walker = self.np_random.choice(walkers)
            dx, dy = self.np_random.choice([(-1,0), (1,0), (0,-1), (0,1)])
            
            nx, ny = walker['x'] + dx, walker['y'] + dy
            if 1 <= nx < self.MAP_WIDTH - 1 and 1 <= ny < self.MAP_HEIGHT - 1:
                if self.map_data[ny][nx] == 1:
                    self.map_data[ny][nx] = 0
                    created_floors += 1
                walkers.append({'x': nx, 'y': ny})

            if len(walkers) > 20:
                walkers.pop(0)

    def _place_entities(self):
        floor_tiles = [(x, y) for y in range(self.MAP_HEIGHT) for x in range(self.MAP_WIDTH) if self.map_data[y][x] == 0]
        self.np_random.shuffle(floor_tiles)

        self.player_pos = list(floor_tiles.pop())
        self.exit_pos = list(floor_tiles.pop())

        self.potions = []
        for _ in range(3):
            if not floor_tiles: break
            self.potions.append({'pos': list(floor_tiles.pop()), 'active': True})

        self.enemies = []
        for _ in range(5):
            if not floor_tiles: break
            spawn_pos = list(floor_tiles.pop())
            
            # Create a 4-tile patrol path if possible
            path = [spawn_pos]
            p = spawn_pos
            moves = [(p[0]+1, p[1]), (p[0]+1, p[1]+1), (p[0], p[1]+1), (p[0], p[1])]
            valid_moves = [m for m in moves if 0 <= m[0] < self.MAP_WIDTH and 0 <= m[1] < self.MAP_HEIGHT and self.map_data[m[1]][m[0]] == 0]
            
            if len(valid_moves) > 2:
                path = [spawn_pos, valid_moves[0], valid_moves[1], valid_moves[2]]

            self.enemies.append({
                'pos': spawn_pos,
                'health': 2,
                'patrol_path': path,
                'patrol_idx': 0
            })

    def _render_game(self):
        # Update and draw particles first (they are in the background)
        self._update_particles()
        for p in self.particles:
            alpha_color = p['color'] + (int(255 * (p['life'] / 15)),)
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), int(p['radius']), alpha_color)
            
        # Prepare dynamic entities for z-sorting
        entities_to_draw = []
        
        # Add player
        entities_to_draw.append({'type': 'player', 'pos': self.player_pos})
        
        # Add active enemies
        for enemy in self.enemies:
            if enemy['health'] > 0:
                entities_to_draw.append({'type': 'enemy', 'pos': enemy['pos']})
        
        # Add active potions
        for potion in self.potions:
            if potion['active']:
                entities_to_draw.append({'type': 'potion', 'pos': potion['pos']})
                
        # Sort entities by Y-coordinate for proper isometric rendering
        entities_to_draw.sort(key=lambda e: self._world_to_screen(e['pos'][0], e['pos'][1])[1])

        # Render map and sorted entities
        entity_idx = 0
        for y in range(self.MAP_HEIGHT):
            for x in range(self.MAP_WIDTH):
                screen_x, screen_y = self._world_to_screen(x, y)
                if self.map_data[y][x] == 0: # Floor
                    points = [
                        (screen_x, screen_y),
                        (screen_x + self.TILE_WIDTH_HALF, screen_y + self.TILE_HEIGHT_HALF),
                        (screen_x, screen_y + self.TILE_HEIGHT),
                        (screen_x - self.TILE_WIDTH_HALF, screen_y + self.TILE_HEIGHT_HALF)
                    ]
                    pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_FLOOR)
                    
                    if [x,y] == self.exit_pos:
                        pygame.gfxdraw.filled_circle(self.screen, screen_x, screen_y + self.TILE_HEIGHT_HALF, 8, self.COLOR_EXIT_GLOW)
                        pygame.gfxdraw.filled_circle(self.screen, screen_x, screen_y + self.TILE_HEIGHT_HALF, 6, self.COLOR_EXIT)

                else: # Wall
                    self._draw_iso_cube(self.screen, x, y, self.COLOR_WALL_TOP, self.COLOR_WALL_SIDE)

                # Draw entities that are on this tile
                while entity_idx < len(entities_to_draw) and self._world_to_screen(entities_to_draw[entity_idx]['pos'][0], entities_to_draw[entity_idx]['pos'][1])[1] <= screen_y + self.TILE_HEIGHT:
                    e = entities_to_draw[entity_idx]
                    ex, ey = self._world_to_screen(e['pos'][0], e['pos'][1])
                    
                    bob = int(math.sin(pygame.time.get_ticks() * 0.002 + ex) * 2)
                    
                    if e['type'] == 'player':
                        pygame.gfxdraw.filled_circle(self.screen, ex, ey - bob, 10, self.COLOR_PLAYER_GLOW)
                        pygame.gfxdraw.filled_circle(self.screen, ex, ey - bob, 8, self.COLOR_PLAYER)
                    elif e['type'] == 'enemy':
                        pygame.gfxdraw.filled_trigon(self.screen, ex, ey-8-bob, ex-8, ey+4-bob, ex+8, ey+4-bob, self.COLOR_ENEMY_GLOW)
                        pygame.gfxdraw.filled_trigon(self.screen, ex, ey-6-bob, ex-6, ey+3-bob, ex+6, ey+3-bob, self.COLOR_ENEMY)
                    elif e['type'] == 'potion':
                        pygame.gfxdraw.filled_circle(self.screen, ex, ey + 4 - bob, 6, self.COLOR_POTION_GLOW)
                        pygame.gfxdraw.filled_circle(self.screen, ex, ey + 4 - bob, 4, self.COLOR_POTION)
                    
                    entity_idx += 1

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1

    def _render_ui(self):
        health_text = self.font.render(f"Health: {self.player_health}", True, self.COLOR_UI_TEXT)
        self.screen.blit(health_text, (10, 10))
        
        score_text = self.font.render(f"Score: {self.score:.1f}", True, self.COLOR_UI_TEXT)
        score_rect = score_text.get_rect(topright=(self.SCREEN_WIDTH - 10, 10))
        self.screen.blit(score_text, score_rect)

        step_text = self.font.render(f"Steps: {self.steps}/1000", True, self.COLOR_UI_TEXT)
        step_rect = step_text.get_rect(topright=(self.SCREEN_WIDTH - 10, 40))
        self.screen.blit(step_text, step_rect)

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
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    pygame.display.set_caption("Isometric Roguelike")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    running = True
    terminated = False
    
    while running:
        action = [0, 0, 0] # Default action: no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and not terminated:
                # Map keyboard presses to a single action for the turn
                if event.key == pygame.K_UP:
                    action[0] = 1
                elif event.key == pygame.K_DOWN:
                    action[0] = 2
                elif event.key == pygame.K_LEFT:
                    action[0] = 3
                elif event.key == pygame.K_RIGHT:
                    action[0] = 4
                elif event.key == pygame.K_SPACE:
                    action[1] = 1
                elif event.key == pygame.K_r: # Reset key
                    obs, info = env.reset()
                    terminated = False
                    continue

                obs, reward, terminated, _, info = env.step(action)
                print(f"Action: {action}, Reward: {reward:.2f}, Score: {info['score']:.2f}, Terminated: {terminated}")

        # Draw the observation from the environment to the screen
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        env.clock.tick(30) # Limit FPS for human play
        
    pygame.quit()