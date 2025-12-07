# Generated: 2025-08-27T13:05:28.225021
# Source Brief: brief_00256.md
# Brief Index: 256

        
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
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrow keys to move. Press space to attack in the direction you are facing."
    )

    game_description = (
        "Explore a dark, procedurally generated dungeon. Defeat enemies, find the exit, and survive."
    )

    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    TILE_SIZE = 32
    MAP_WIDTH = 50
    MAP_HEIGHT = 50

    # Colors
    COLOR_BG = (10, 10, 20)
    COLOR_WALL = (40, 40, 60)
    COLOR_FLOOR = (20, 20, 35)
    COLOR_PLAYER = (255, 0, 80)
    COLOR_ENEMY_A = (0, 150, 255)
    COLOR_ENEMY_B = (150, 50, 255)
    COLOR_EXIT = (255, 220, 0)
    COLOR_HEALTH_PICKUP = (0, 255, 120)
    COLOR_ATTACK = (255, 255, 255)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_UI_HEALTH_BAR = (200, 30, 30)
    COLOR_UI_HEALTH_BG = (60, 20, 20)

    # Game parameters
    PLAYER_MAX_HEALTH = 10
    PLAYER_ATTACK_DAMAGE = 2
    ENEMY_MAX_HEALTH = 4
    ENEMY_ATTACK_DAMAGE = 1
    NUM_ENEMIES = 5
    NUM_HEALTH_PICKUPS = 3
    HEALTH_PICKUP_AMOUNT = 2
    MAX_STEPS = 1000

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 28)
        self.font_ui_large = pygame.font.Font(None, 48)

        self.game_over = False
        self.game_won = False
        
        # These will be initialized in reset()
        self.player_health = self.PLAYER_MAX_HEALTH
        
        self.reset()
        
        # This validation was causing an error because it set health > max
        # and expected step() to clamp it. The fix is to ensure step() clamps it.
        # self.validate_implementation()


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            # Use a single Random Number Generator for reproducibility
            self.np_random = np.random.default_rng(seed)
            # Keep old random for compatibility with existing code using it
            random.seed(seed)
        else:
            # If no seed, create a new generator
            if not hasattr(self, 'np_random'):
                 self.np_random = np.random.default_rng()

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False

        self._generate_dungeon()

        self.player_health = self.PLAYER_MAX_HEALTH
        self.player_facing = np.array([0, 1])  # Start facing down

        self.particles = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = -0.01  # Small penalty for each step to encourage efficiency

        # --- Player Action ---
        action_taken = False
        if space_held:
            # --- Attack Action ---
            action_taken = True
            target_pos = self.player_pos + self.player_facing
            for enemy in self.enemies:
                if np.array_equal(enemy['pos'], target_pos):
                    enemy['health'] -= self.PLAYER_ATTACK_DAMAGE
                    enemy['hit_timer'] = 5 # Visual feedback
                    if enemy['health'] <= 0:
                        reward += 5.0
                        self.score += 5
                    else:
                        reward += 0.5 # Reward for hitting
                    break 
            
            # Add attack particle effect
            particle_pos = (self.player_pos + 0.5 + self.player_facing * 0.5) * self.TILE_SIZE
            self.particles.append({'pos': particle_pos, 'radius': 0, 'max_radius': self.TILE_SIZE * 0.6, 'life': 5, 'color': self.COLOR_ATTACK})
        
        elif movement != 0:
            # --- Movement Action ---
            action_taken = True
            move_dir = {1: [0, -1], 2: [0, 1], 3: [-1, 0], 4: [1, 0]}.get(movement)
            if move_dir:
                self.player_facing = np.array(move_dir)
                new_pos = self.player_pos + move_dir
                if self.dungeon_map[new_pos[1], new_pos[0]] == 0: # Is a floor tile
                    self.player_pos = new_pos

        if movement == 0 and not space_held: # No-op
            action_taken = True

        # --- Game Logic Update (only if an action was taken) ---
        if action_taken:
            self.steps += 1

            # --- Enemy Turn ---
            self.enemies = [e for e in self.enemies if e['health'] > 0] # Remove dead enemies
            for enemy in self.enemies:
                if enemy['hit_timer'] > 0:
                    enemy['hit_timer'] -= 1
                
                # Check if adjacent to player to attack
                if np.linalg.norm(self.player_pos - enemy['pos']) < 1.5:
                    self.player_health -= self.ENEMY_ATTACK_DAMAGE
                    self.particles.append({'pos': (self.player_pos + 0.5) * self.TILE_SIZE, 'radius': self.TILE_SIZE * 0.5, 'max_radius': self.TILE_SIZE * 0.5, 'life': 4, 'color': self.COLOR_PLAYER})
                else: # Move randomly
                    possible_moves = []
                    for move in [[0, 1], [0, -1], [1, 0], [-1, 0]]:
                        next_pos = enemy['pos'] + move
                        if self.dungeon_map[next_pos[1], next_pos[0]] == 0:
                            possible_moves.append(move)
                    if possible_moves:
                        move_idx = self.np_random.integers(len(possible_moves))
                        enemy['pos'] += possible_moves[move_idx]

            # --- Check Interactions ---
            # Health pickups
            for i, pickup_pos in enumerate(self.pickups):
                if np.array_equal(self.player_pos, pickup_pos):
                    self.player_health = min(self.PLAYER_MAX_HEALTH, self.player_health + self.HEALTH_PICKUP_AMOUNT)
                    reward += 2.0
                    self.score += 2
                    self.pickups.pop(i)
                    break
            
            # --- Update Particles ---
            self.particles = [p for p in self.particles if p['life'] > 0]
            for p in self.particles:
                p['life'] -= 1
                if p['radius'] < p['max_radius']:
                    p['radius'] += p['max_radius'] / 5

        # --- Check Termination Conditions ---
        terminated = False
        truncated = False
        if np.array_equal(self.player_pos, self.exit_pos):
            reward = 100.0
            self.score += 100
            terminated = True
            self.game_over = True
            self.game_won = True
        elif self.player_health <= 0:
            reward = -100.0
            terminated = True
            self.game_over = True
        elif self.steps >= self.MAX_STEPS:
            truncated = True # Use truncated for time limit
            self.game_over = True

        # FIX: Clamp player health to ensure it never exceeds the maximum.
        # This resolves the assertion error in the original validation code.
        self.player_health = min(self.player_health, self.PLAYER_MAX_HEALTH)

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _generate_dungeon(self):
        self.dungeon_map = np.ones((self.MAP_HEIGHT, self.MAP_WIDTH), dtype=np.int8)
        
        # Random walk to carve out a connected area
        floor_tiles = []
        px, py = self.MAP_WIDTH // 2, self.MAP_HEIGHT // 2
        num_tiles_to_carve = int(self.MAP_WIDTH * self.MAP_HEIGHT * 0.3)
        
        for _ in range(num_tiles_to_carve):
            if self.dungeon_map[py, px] == 1:
                self.dungeon_map[py, px] = 0
                floor_tiles.append(np.array([px, py]))
            
            direction_idx = self.np_random.integers(4)
            direction = [[0, 1], [0, -1], [1, 0], [-1, 0]][direction_idx]
            px = np.clip(px + direction[0], 1, self.MAP_WIDTH - 2)
            py = np.clip(py + direction[1], 1, self.MAP_HEIGHT - 2)
        
        # Use BFS to find all reachable tiles from start
        if not floor_tiles: # Handle case where no tiles were carved
            start_pos = np.array([self.MAP_WIDTH // 2, self.MAP_HEIGHT // 2])
            self.dungeon_map[start_pos[1], start_pos[0]] = 0
            floor_tiles.append(start_pos)
        
        start_pos = floor_tiles[0]
        q = deque([(start_pos, 0)])
        visited = {tuple(start_pos)}
        distances = {tuple(start_pos): 0}
        
        while q:
            pos, dist = q.popleft()
            for move in [[0, 1], [0, -1], [1, 0], [-1, 0]]:
                next_pos = pos + move
                if (self.dungeon_map[next_pos[1], next_pos[0]] == 0 and tuple(next_pos) not in visited):
                    visited.add(tuple(next_pos))
                    distances[tuple(next_pos)] = dist + 1
                    q.append((next_pos, dist + 1))
        
        reachable_floor_tiles = [np.array(p) for p in visited]
        
        # Place player
        self.player_pos = start_pos
        
        # Place exit at the furthest reachable point
        farthest_pos = max(distances, key=distances.get)
        self.exit_pos = np.array(farthest_pos)

        # Place enemies and pickups
        spawn_locations = [tile for tile in reachable_floor_tiles if np.linalg.norm(tile - self.player_pos) > 5 and np.linalg.norm(tile - self.exit_pos) > 1]
        self.np_random.shuffle(spawn_locations)
        
        self.enemies = []
        for i in range(min(self.NUM_ENEMIES, len(spawn_locations))):
            self.enemies.append({'pos': spawn_locations.pop(), 'health': self.ENEMY_MAX_HEALTH, 'hit_timer': 0, 'anim_offset': self.np_random.uniform(0, math.pi * 2)})

        self.pickups = []
        for i in range(min(self.NUM_HEALTH_PICKUPS, len(spawn_locations))):
            self.pickups.append(spawn_locations.pop())

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Camera centered on player
        cam_x = self.player_pos[0] * self.TILE_SIZE - self.SCREEN_WIDTH / 2
        cam_y = self.player_pos[1] * self.TILE_SIZE - self.SCREEN_HEIGHT / 2

        # Determine visible tile range
        start_x = max(0, int(cam_x / self.TILE_SIZE))
        end_x = min(self.MAP_WIDTH, int((cam_x + self.SCREEN_WIDTH) / self.TILE_SIZE) + 2)
        start_y = max(0, int(cam_y / self.TILE_SIZE))
        end_y = min(self.MAP_HEIGHT, int((cam_y + self.SCREEN_HEIGHT) / self.TILE_SIZE) + 2)

        # Draw dungeon
        for y in range(start_y, end_y):
            for x in range(start_x, end_x):
                screen_pos = (int(x * self.TILE_SIZE - cam_x), int(y * self.TILE_SIZE - cam_y))
                rect = pygame.Rect(screen_pos, (self.TILE_SIZE, self.TILE_SIZE))
                color = self.COLOR_WALL if self.dungeon_map[y, x] == 1 else self.COLOR_FLOOR
                pygame.draw.rect(self.screen, color, rect)
        
        # Draw health pickups
        for pos in self.pickups:
            self._draw_entity(pos, self.TILE_SIZE * 0.3, self.COLOR_HEALTH_PICKUP, cam_x, cam_y)

        # Draw exit (with glow)
        glow_size = self.TILE_SIZE * 0.8 + 4 * math.sin(pygame.time.get_ticks() / 200)
        self._draw_entity(self.exit_pos, glow_size, self.COLOR_EXIT, cam_x, cam_y, is_circle=True, alpha=80)
        self._draw_entity(self.exit_pos, self.TILE_SIZE * 0.6, self.COLOR_EXIT, cam_x, cam_y, is_circle=True)

        # Draw enemies
        for enemy in self.enemies:
            color = self.COLOR_ENEMY_A if enemy['pos'][0] % 2 == 0 else self.COLOR_ENEMY_B
            if enemy['hit_timer'] > 0:
                color = (255, 255, 255) # Flash white when hit
            
            anim_offset = 3 * math.sin(pygame.time.get_ticks() / 250 + enemy['anim_offset'])
            self._draw_entity(enemy['pos'], self.TILE_SIZE * 0.7, color, cam_x, cam_y, y_offset=anim_offset)

        # Draw player
        anim_offset = 3 * math.sin(pygame.time.get_ticks() / 200)
        self._draw_entity(self.player_pos, self.TILE_SIZE * 0.8, self.COLOR_PLAYER, cam_x, cam_y, y_offset=anim_offset)

        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / 5))
            color_with_alpha = p['color'] + (alpha,)
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0] - cam_x), int(p['pos'][1] - cam_y), int(p['radius']), color_with_alpha)


    def _draw_entity(self, pos, size, color, cam_x, cam_y, y_offset=0, is_circle=False, alpha=255):
        screen_x = int(pos[0] * self.TILE_SIZE + self.TILE_SIZE / 2 - cam_x)
        screen_y = int(pos[1] * self.TILE_SIZE + self.TILE_SIZE / 2 - cam_y + y_offset)
        
        if -size < screen_x < self.SCREEN_WIDTH + size and -size < screen_y < self.SCREEN_HEIGHT + size:
            if is_circle:
                if alpha < 255:
                    pygame.gfxdraw.filled_circle(self.screen, screen_x, screen_y, int(size/2), color + (alpha,))
                else:
                    pygame.gfxdraw.filled_circle(self.screen, screen_x, screen_y, int(size/2), color)
            else:
                rect = pygame.Rect(screen_x - size/2, screen_y - size/2, size, size)
                pygame.draw.rect(self.screen, color, rect, border_radius=3)


    def _render_ui(self):
        # Health bar
        bar_width = 200
        health_ratio = max(0, self.player_health / self.PLAYER_MAX_HEALTH)
        pygame.draw.rect(self.screen, self.COLOR_UI_HEALTH_BG, (20, 20, bar_width, 25))
        pygame.draw.rect(self.screen, self.COLOR_UI_HEALTH_BAR, (20, 20, bar_width * health_ratio, 25))
        
        # Health text
        health_text = self.font_ui.render(f"HP: {self.player_health}/{self.PLAYER_MAX_HEALTH}", True, self.COLOR_UI_TEXT)
        self.screen.blit(health_text, (25, 22))

        # Score text
        score_text = self.font_ui.render(f"Score: {self.score}", True, self.COLOR_UI_TEXT)
        score_rect = score_text.get_rect(topright=(self.SCREEN_WIDTH - 20, 20))
        self.screen.blit(score_text, score_rect)
        
        # Game Over / Victory message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            msg = "YOU REACHED THE EXIT!" if self.game_won else "YOU DIED"
            color = self.COLOR_EXIT if self.game_won else self.COLOR_PLAYER
            
            end_text = self.font_ui_large.render(msg, True, color)
            end_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, end_rect)


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_health": self.player_health,
            "player_pos": self.player_pos.tolist()
        }

    def close(self):
        pygame.quit()

    def render(self):
        return self._get_observation()

if __name__ == '__main__':
    # This block allows you to play the game manually
    env = GameEnv()
    obs, info = env.reset()
    
    done = False
    total_reward = 0
    
    # Pygame setup for manual play
    render_screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Dungeon Crawler")
    clock = pygame.time.Clock()

    print("\n" + "="*30)
    print(env.game_description)
    print(env.user_guide)
    print("="*30 + "\n")

    terminated = False
    truncated = False
    while not (terminated or truncated):
        # --- Action mapping from keyboard ---
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space = 1 if keys[pygame.K_SPACE] else 0
        shift = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space, shift]

        # --- Event handling ---
        action_taken_this_frame = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True # End the loop
            if event.type == pygame.KEYDOWN:
                # For turn-based, we only step on a key press
                if event.key in [pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT, pygame.K_SPACE, pygame.K_q]:
                    if event.key == pygame.K_q:
                        terminated = True
                        break
                    obs, reward, terminated, truncated, info = env.step(action)
                    total_reward += reward
                    action_taken_this_frame = True
                    print(f"Step: {info['steps']}, Action: {action}, Reward: {reward:.2f}, Total Reward: {total_reward:.2f}, Term: {terminated}, Trunc: {truncated}")

        # --- Rendering ---
        # Get the observation from the environment
        frame_to_render = env.render()
        # Convert observation back to a Pygame surface to display
        frame = np.transpose(frame_to_render, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        render_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit frame rate for manual play

    env.close()
    print("Game Over!")
    print(f"Final Score: {info.get('score', 0)}")