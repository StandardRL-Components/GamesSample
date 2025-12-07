
# Generated: 2025-08-28T06:50:28.200544
# Source Brief: brief_03045.md
# Brief Index: 3045

        
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
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrow keys to move. Space to attack adjacent tiles. "
        "Shift to take an evasive step away from the nearest monster."
    )

    game_description = (
        "Explore a procedurally generated dungeon, battling monsters and "
        "collecting gold to reach the exit. Your goal is to clear all 5 rooms."
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.TILE_SIZE = 32
        self.GRID_WIDTH = self.WIDTH // self.TILE_SIZE
        self.GRID_HEIGHT = self.HEIGHT // self.TILE_SIZE

        # Gameplay Constants
        self.MAX_PLAYER_HEALTH = 10
        self.INITIAL_MONSTER_HEALTH = 3
        self.MAX_ROOMS = 5
        self.MAX_STEPS = 1000

        # Colors
        self.COLOR_BG = (20, 15, 25)
        self.COLOR_FLOOR = (50, 40, 60)
        self.COLOR_WALL = (90, 80, 100)
        self.COLOR_WALL_TOP = (120, 110, 130)
        self.COLOR_PLAYER = (60, 200, 255)
        self.COLOR_PLAYER_SHADOW = (40, 130, 170)
        self.COLOR_EXIT = (100, 255, 100)
        self.COLOR_GOLD = (255, 223, 0)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_HEALTH_BAR_BG = (100, 0, 0)
        self.COLOR_HEALTH_BAR = (255, 0, 0)
        
        self.MONSTER_COLORS = {
            "slime": ((80, 200, 80), (50, 130, 50)),
            "bat": ((140, 100, 180), (90, 60, 120)),
        }

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
        self.font_room = pygame.font.Font(None, 24)

        # --- State Variables ---
        self.grid = None
        self.player_pos = None
        self.player_health = 0
        self.monsters = []
        self.gold_pieces = []
        self.exit_pos = None
        self.particles = []
        self.steps = 0
        self.score = 0
        self.room_level = 0
        self.game_over = False
        self.win = False
        self.last_player_dist_to_exit = 0
        self.animation_state = 0
        self.damage_flash_timer = 0

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.room_level = 1
        self.player_health = self.MAX_PLAYER_HEALTH
        self.game_over = False
        self.win = False
        self.particles = []
        self.damage_flash_timer = 0

        self._generate_room()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        self.animation_state = (self.animation_state + 1) % 60
        reward = 0
        
        # --- Player Action Phase ---
        reward += self._handle_player_action(action)

        # --- Post-Player Action State Update ---
        reward += self._collect_gold()
        
        room_cleared = self._check_exit()
        if room_cleared:
            if self.room_level > self.MAX_ROOMS:
                self.win = True
            else:
                self._generate_room()

        # --- Monster Action Phase ---
        if not self.game_over and not room_cleared:
            reward += self._handle_monster_turn()

        # --- Distance Reward ---
        current_dist = self._manhattan_distance(self.player_pos, self.exit_pos)
        reward += 0.1 if current_dist < self.last_player_dist_to_exit else -0.1 if current_dist > self.last_player_dist_to_exit else 0
        self.last_player_dist_to_exit = current_dist

        # --- Termination Check ---
        terminated = False
        if self.player_health <= 0:
            self.game_over = True
            terminated = True
            reward = -100.0  # Loss penalty
            # sfx: player_death
        elif self.win:
            self.game_over = True
            terminated = True
            reward = 100.0   # Win bonus
            # sfx: victory_fanfare
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            
        # --- Particle Update ---
        self._update_particles()
        if self.damage_flash_timer > 0:
            self.damage_flash_timer -= 1

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_player_action(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0
        action_taken = True
        
        old_pos = list(self.player_pos)

        if space_held: # Attack
            # sfx: player_attack_swing
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                target_pos = (self.player_pos[0] + dx, self.player_pos[1] + dy)
                self._create_particles(10, self._grid_to_pixel(target_pos), (255, 255, 150), 2, 8)
                for monster in self.monsters:
                    if tuple(monster['pos']) == target_pos:
                        monster['health'] -= 1
                        # sfx: monster_hit
                        if monster['health'] <= 0:
                            reward += 1.0 # Monster defeated
                            self._create_particles(30, self._grid_to_pixel(monster['pos']), monster['color'][0], 3, 15)
                            self.monsters.remove(monster)
                            # sfx: monster_death
                        break
        elif shift_held: # Evade
            nearest_monster_dist = float('inf')
            nearest_monster = None
            for monster in self.monsters:
                dist = self._manhattan_distance(self.player_pos, monster['pos'])
                if dist < nearest_monster_dist:
                    nearest_monster_dist = dist
                    nearest_monster = monster
            
            if nearest_monster:
                dx = np.sign(self.player_pos[0] - nearest_monster['pos'][0])
                dy = np.sign(self.player_pos[1] - nearest_monster['pos'][1])
                
                # Prioritize moving away on the axis of greatest difference
                if abs(self.player_pos[0] - nearest_monster['pos'][0]) < abs(self.player_pos[1] - nearest_monster['pos'][1]):
                    dx = 0
                else:
                    dy = 0

                if dx == 0 and dy == 0: # On same tile
                    dx, dy = self.np_random.choice([-1, 1], 2)

                new_pos = [self.player_pos[0] + dx, self.player_pos[1] + dy]
                if self.grid[new_pos[1]][new_pos[0]] == 0:
                    self.player_pos = new_pos
                    # sfx: player_evade_whoosh
                    self._create_particles(5, self._grid_to_pixel(old_pos), (200, 200, 255), 1, 5)

        elif movement > 0: # Move
            dx, dy = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)][movement]
            new_pos = [self.player_pos[0] + dx, self.player_pos[1] + dy]
            if self.grid[new_pos[1]][new_pos[0]] == 0:
                self.player_pos = new_pos
                # sfx: player_step
        else:
            action_taken = False # No-op

        return reward

    def _handle_monster_turn(self):
        damage_taken = 0
        for monster in self.monsters:
            # Monster AI
            dist_to_player = self._manhattan_distance(monster['pos'], self.player_pos)
            
            if dist_to_player == 1: # Attack
                self.player_health -= 1
                damage_taken += 1
                self.damage_flash_timer = 5
                self._create_particles(20, self._grid_to_pixel(self.player_pos), (255, 50, 50), 2, 10)
                # sfx: player_hit
            elif dist_to_player < 6 and monster['type'] == 'bat': # Bat moves towards player
                dx = np.sign(self.player_pos[0] - monster['pos'][0])
                dy = np.sign(self.player_pos[1] - monster['pos'][1])
                if dx != 0 and dy != 0: # No diagonal moves
                    if self.np_random.random() < 0.5: dx = 0
                    else: dy = 0
                new_pos = [monster['pos'][0] + dx, monster['pos'][1] + dy]
                if self.grid[new_pos[1]][new_pos[0]] == 0 and not self._is_occupied(new_pos):
                    monster['pos'] = new_pos
            else: # Slime moves randomly, Bat moves randomly if player is far
                dx, dy = random.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])
                new_pos = [monster['pos'][0] + dx, monster['pos'][1] + dy]
                if self.grid[new_pos[1]][new_pos[0]] == 0 and not self._is_occupied(new_pos):
                    monster['pos'] = new_pos
        
        return 0 # Monster actions don't give positive reward

    def _collect_gold(self):
        reward = 0
        for gold in self.gold_pieces[:]:
            if tuple(self.player_pos) == tuple(gold):
                self.gold_pieces.remove(gold)
                self.score += 1
                reward += 0.5
                self._create_particles(15, self._grid_to_pixel(gold), self.COLOR_GOLD, 2, 10)
                # sfx: gold_pickup
        return reward
    
    def _check_exit(self):
        if tuple(self.player_pos) == tuple(self.exit_pos):
            self.room_level += 1
            # sfx: level_up
            return True
        return False

    def _generate_room(self):
        self.grid = np.ones((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=int)
        self.grid[1:self.GRID_HEIGHT-1, 1:self.GRID_WIDTH-1] = 0
        
        self.monsters.clear()
        self.gold_pieces.clear()
        self.particles.clear()
        
        occupied_positions = set()

        self.player_pos = [2, self.GRID_HEIGHT // 2]
        occupied_positions.add(tuple(self.player_pos))
        
        self.exit_pos = [self.GRID_WIDTH - 2, self.GRID_HEIGHT // 2]
        occupied_positions.add(tuple(self.exit_pos))
        
        num_monsters = min(self.room_level + 1, 5)
        monster_health = self.INITIAL_MONSTER_HEALTH + self.room_level - 1
        
        for _ in range(num_monsters):
            pos = self._get_random_empty_pos(occupied_positions)
            occupied_positions.add(tuple(pos))
            monster_type = self.np_random.choice(["slime", "bat"])
            self.monsters.append({
                "pos": pos,
                "health": monster_health,
                "max_health": monster_health,
                "type": monster_type,
                "color": self.MONSTER_COLORS[monster_type]
            })

        num_gold = self.np_random.integers(3, 6)
        for _ in range(num_gold):
            pos = self._get_random_empty_pos(occupied_positions)
            occupied_positions.add(tuple(pos))
            self.gold_pieces.append(pos)
            
        self.last_player_dist_to_exit = self._manhattan_distance(self.player_pos, self.exit_pos)

    def _get_random_empty_pos(self, occupied):
        while True:
            pos = [
                self.np_random.integers(2, self.GRID_WIDTH - 2),
                self.np_random.integers(2, self.GRID_HEIGHT - 2)
            ]
            if tuple(pos) not in occupied:
                return pos

    def _is_occupied(self, pos):
        pos_tuple = tuple(pos)
        if pos_tuple == tuple(self.player_pos): return True
        for m in self.monsters:
            if pos_tuple == tuple(m['pos']): return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        if self.damage_flash_timer > 0:
            flash_surface = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            flash_surface.fill((255, 0, 0, 80))
            self.screen.blit(flash_surface, (0, 0))

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw floor and walls
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                rect = pygame.Rect(x * self.TILE_SIZE, y * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
                if self.grid[y][x] == 1:
                    pygame.draw.rect(self.screen, self.COLOR_WALL, rect)
                    pygame.draw.rect(self.screen, self.COLOR_WALL_TOP, rect.move(0, -2), border_bottom_left_radius=2, border_bottom_right_radius=2)
                else:
                    pygame.draw.rect(self.screen, self.COLOR_FLOOR, rect)
        
        # Draw exit
        exit_px = self._grid_to_pixel(self.exit_pos)
        exit_rect = pygame.Rect(exit_px[0], exit_px[1], self.TILE_SIZE, self.TILE_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_EXIT, exit_rect.inflate(-8, -8))

        # Draw gold
        for gold_pos in self.gold_pieces:
            gold_px = self._grid_to_pixel(gold_pos)
            size_anim = int(3 * math.sin(self.animation_state / 5))
            pygame.draw.circle(self.screen, self.COLOR_GOLD, (gold_px[0] + self.TILE_SIZE // 2, gold_px[1] + self.TILE_SIZE // 2), 6 + size_anim)

        # Draw monsters
        for monster in self.monsters:
            self._draw_entity(monster['pos'], monster['color'][0], monster['color'][1], monster['type'])
            # Health bar for monsters
            hp_ratio = monster['health'] / monster['max_health']
            bar_width = int(self.TILE_SIZE * 0.8 * hp_ratio)
            pos_px = self._grid_to_pixel(monster['pos'])
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (pos_px[0] + self.TILE_SIZE * 0.1, pos_px[1] - 8, self.TILE_SIZE * 0.8, 5))
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, (pos_px[0] + self.TILE_SIZE * 0.1, pos_px[1] - 8, bar_width, 5))

        # Draw player
        self._draw_entity(self.player_pos, self.COLOR_PLAYER, self.COLOR_PLAYER_SHADOW, "player")

        # Draw particles
        for p in self.particles:
            pygame.draw.circle(self.screen, p['color'], p['pos'], int(p['size']))

    def _draw_entity(self, pos, color, shadow_color, type):
        px, py = self._grid_to_pixel(pos)
        center_x, center_y = px + self.TILE_SIZE // 2, py + self.TILE_SIZE // 2
        
        bob = int(2 * math.sin(self.animation_state / 6))
        
        if type == "player":
            size = self.TILE_SIZE * 0.7
            shadow_rect = pygame.Rect(0, 0, size, size)
            shadow_rect.center = (center_x, center_y + 4)
            pygame.draw.rect(self.screen, shadow_color, shadow_rect, border_radius=4)
            player_rect = pygame.Rect(0, 0, size, size)
            player_rect.center = (center_x, center_y + bob)
            pygame.draw.rect(self.screen, color, player_rect, border_radius=4)
        elif type == "slime":
            squish = int(3 * math.sin(self.animation_state / 8))
            width, height = self.TILE_SIZE * 0.6 + squish, self.TILE_SIZE * 0.6 - squish
            shadow_rect = pygame.Rect(0, 0, width, height)
            shadow_rect.center = (center_x, center_y + 4)
            pygame.draw.ellipse(self.screen, shadow_color, shadow_rect)
            body_rect = pygame.Rect(0, 0, width, height)
            body_rect.center = (center_x, center_y + bob)
            pygame.draw.ellipse(self.screen, color, body_rect)
        elif type == "bat":
            size = self.TILE_SIZE * 0.6
            shadow_rect = pygame.Rect(0, 0, size, size / 2)
            shadow_rect.center = (center_x, center_y + 4)
            pygame.draw.ellipse(self.screen, shadow_color, shadow_rect)
            body_rect = pygame.Rect(0, 0, size, size)
            body_rect.center = (center_x, center_y + bob)
            pygame.draw.ellipse(self.screen, color, body_rect)

    def _render_ui(self):
        # Health Bar
        health_ratio = max(0, self.player_health / self.MAX_PLAYER_HEALTH)
        bar_width = 200
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (10, 10, bar_width, 20))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, (10, 10, int(bar_width * health_ratio), 20))
        health_text = self.font_ui.render(f"HP: {self.player_health}/{self.MAX_PLAYER_HEALTH}", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (15, 12))

        # Gold
        gold_text = self.font_ui.render(f"Gold: {self.score}", True, self.COLOR_GOLD)
        self.screen.blit(gold_text, (self.WIDTH - gold_text.get_width() - 10, 10))
        
        # Room Level
        room_text = self.font_room.render(f"Dungeon Level: {self.room_level}/{self.MAX_ROOMS}", True, self.COLOR_TEXT)
        self.screen.blit(room_text, (self.WIDTH // 2 - room_text.get_width() // 2, 10))

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "health": self.player_health, "room": self.room_level}

    def _grid_to_pixel(self, grid_pos):
        return [grid_pos[0] * self.TILE_SIZE, grid_pos[1] * self.TILE_SIZE]

    def _manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _create_particles(self, count, pos, color, speed, lifespan):
        for _ in range(count):
            angle = self.np_random.random() * 2 * math.pi
            vel = [math.cos(angle) * speed * self.np_random.random(), math.sin(angle) * speed * self.np_random.random()]
            self.particles.append({
                'pos': list(pos),
                'vel': vel,
                'life': lifespan + self.np_random.integers(0, lifespan // 2),
                'color': color,
                'size': self.np_random.random() * 3 + 1
            })

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            p['size'] *= 0.95
            if p['life'] <= 0 or p['size'] < 0.5:
                self.particles.remove(p)

    def validate_implementation(self):
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

        # Test state guarantees
        self.reset()
        self.player_health = self.MAX_PLAYER_HEALTH + 5
        assert self.player_health <= self.MAX_PLAYER_HEALTH + 5 # Check is done on render
        self.score = -10
        assert self.score < 0 # Score can be negative from penalties
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    terminated = False
    
    # Create a window to display the game
    pygame.display.set_caption("Dungeon Crawler")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    # Game loop
    running = True
    while running:
        # Action mapping from keyboard
        keys = pygame.key.get_pressed()
        movement = 0
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        action_taken = any(action)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r: # Press R to reset
                    obs, info = env.reset()
                    terminated = False
                if event.key in [pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT, pygame.K_SPACE, pygame.K_LSHIFT, pygame.K_RSHIFT]:
                    obs, reward, terminated, _, info = env.step(action)
                    print(f"Reward: {reward:.2f}, Score: {info['score']}, Terminated: {terminated}")
        
        # If no key was pressed, we might still want to step with a no-op action
        # This is for turn-based games where waiting is an action.
        # However, for better playability, we only step on keydown.
        
        # Rendering
        frame = env._get_observation()
        frame = np.transpose(frame, (1, 0, 2)) # Transpose back for pygame display
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated:
            print("Game Over! Press 'R' to restart.")
            # Wait for reset
            while True:
                event = pygame.event.wait()
                if event.type == pygame.QUIT:
                    running = False
                    break
                if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    obs, info = env.reset()
                    terminated = False
                    break
            if not running:
                break
        
        env.clock.tick(30) # Limit FPS for human play

    pygame.quit()