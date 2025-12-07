
# Generated: 2025-08-28T04:43:30.387392
# Source Brief: brief_02397.md
# Brief Index: 2397

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
from collections import deque
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Use arrow keys for movement. Press Space to attack adjacent enemies. "
        "If you don't move or attack, you will defend (50% damage reduction)."
    )

    game_description = (
        "A turn-based dungeon crawler. Explore a procedurally generated maze, "
        "fight monsters to gain experience, and find the exit to escape."
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.TILE_SIZE = 32
        self.MAX_STEPS = 1000

        # --- Colors ---
        self.COLOR_BG = (20, 20, 20)
        self.COLOR_FLOOR = (80, 60, 40)
        self.COLOR_WALL = (130, 110, 90)
        self.COLOR_HERO = (50, 200, 50)
        self.COLOR_ENEMY = (200, 50, 50)
        self.COLOR_EXIT = (50, 150, 250)
        self.COLOR_XP = (255, 215, 0)
        self.COLOR_WHITE = (255, 255, 255)
        self.COLOR_BLACK = (0, 0, 0)
        self.COLOR_HEALTH_FG = (220, 20, 60)
        self.COLOR_HEALTH_BG = (139, 0, 0)
        self.COLOR_DEFEND = (100, 100, 255, 150) # Semi-transparent blue for defend shield

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont('monospace', 16, bold=True)
        self.font_large = pygame.font.SysFont('monospace', 24, bold=True)
        
        # --- State Variables ---
        self._np_random = None
        self.dungeon = None
        self.dungeon_width = 0
        self.dungeon_height = 0
        self.start_pos = None
        self.exit_pos = None
        self.hero = None
        self.enemies = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.action_feedback = [] # To store visual effects for one frame

        self.validate_implementation()
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._np_random, _ = gym.utils.seeding.np_random(seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.action_feedback.clear()
        
        self.dungeon_level = 1
        self._generate_dungeon(width=31, height=21) # Odd numbers work best for maze gen
        self._place_entities()
        
        self.last_hero_dist_to_exit = self._manhattan_distance(self.hero['pos'], self.exit_pos)

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.action_feedback.clear()
        reward = 0.0
        is_defending = False

        movement, space_held, shift_held = action
        
        # --- Player Turn ---
        # Action Priority: Attack > Move > Defend
        if space_held:
            reward += self._handle_attack()
        elif movement != 0:
            reward += self._handle_movement(movement)
        else:
            is_defending = True
            reward -= 0.01 # Small penalty for inaction
            self.action_feedback.append({'type': 'defend', 'pos': self.hero['pos']})
            # sound: player_defend.wav

        # --- Enemy Turn ---
        for enemy in self.enemies:
            if self._manhattan_distance(self.hero['pos'], enemy['pos']) == 1:
                damage = enemy['attack']
                if is_defending:
                    damage *= 0.5
                self.hero['hp'] -= damage
                self.action_feedback.append({'type': 'hit', 'pos': self.hero['pos'], 'color': self.COLOR_ENEMY})
                # sound: hero_hurt.wav
        
        self.steps += 1

        # --- Check Termination Conditions ---
        terminated = False
        if self.hero['hp'] <= 0:
            self.hero['hp'] = 0
            terminated = True
            self.game_over = True
            reward = -100.0
            # sound: game_over.wav
        
        if self.hero['pos'] == self.exit_pos:
            terminated = True
            self.game_over = True
            reward += 100.0
            self.score += 100
            # sound: level_complete.wav

        if self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_attack(self):
        reward = 0
        # Find nearest enemy
        target_enemy = None
        min_dist = float('inf')
        for enemy in self.enemies:
            dist = self._manhattan_distance(self.hero['pos'], enemy['pos'])
            if dist < min_dist:
                min_dist = dist
                target_enemy = enemy
        
        # Attack if adjacent
        if target_enemy and min_dist == 1:
            # sound: player_attack.wav
            target_enemy['hp'] -= self.hero['attack']
            self.action_feedback.append({'type': 'hit', 'pos': target_enemy['pos'], 'color': self.COLOR_WHITE})
            
            if target_enemy['hp'] <= 0:
                self.enemies.remove(target_enemy)
                self.hero['xp'] += 5
                reward += 1.0
                self.score += 10
                # sound: enemy_die.wav
                if self.hero['xp'] >= self.hero['max_xp']:
                    self._level_up_hero()
        return reward

    def _handle_movement(self, movement):
        dx, dy = 0, 0
        if movement == 1: dy = -1  # Up
        elif movement == 2: dy = 1   # Down
        elif movement == 3: dx = -1  # Left
        elif movement == 4: dx = 1   # Right
        
        new_pos = (self.hero['pos'][0] + dx, self.hero['pos'][1] + dy)
        
        if self.dungeon[new_pos[1]][new_pos[0]] == 0: # 0 is floor
            self.hero['pos'] = new_pos
            # sound: player_move.wav
            
            # Reward for getting closer to exit
            new_dist = self._manhattan_distance(self.hero['pos'], self.exit_pos)
            reward = (self.last_hero_dist_to_exit - new_dist) * 0.1
            self.last_hero_dist_to_exit = new_dist
            return reward
        return 0

    def _level_up_hero(self):
        self.hero['xp'] = 0
        self.hero['max_xp'] *= 1.5
        self.hero['max_hp'] += 5
        self.hero['hp'] = self.hero['max_hp'] # Full heal on level up
        self.hero['attack'] += 1
        self.action_feedback.append({'type': 'levelup', 'pos': self.hero['pos']})
        # sound: level_up.wav

    def _generate_dungeon(self, width, height):
        self.dungeon_width = width
        self.dungeon_height = height
        self.dungeon = np.ones((height, width), dtype=np.uint8) # 1 = wall
        
        # Recursive backtracker algorithm
        stack = deque()
        start_x, start_y = (
            self._np_random.integers(0, width // 2) * 2 + 1,
            self._np_random.integers(0, height // 2) * 2 + 1,
        )
        self.start_pos = (start_x, start_y)
        self.dungeon[start_y, start_x] = 0
        stack.append(self.start_pos)
        
        visited_cells = {self.start_pos}
        
        while stack:
            cx, cy = stack[-1]
            neighbors = []
            for dx, dy in [(0, -2), (0, 2), (-2, 0), (2, 0)]:
                nx, ny = cx + dx, cy + dy
                if 0 < nx < width - 1 and 0 < ny < height - 1 and (nx, ny) not in visited_cells:
                    neighbors.append((nx, ny))
            
            if neighbors:
                # Use the seeded RNG for neighbor selection
                idx = self._np_random.integers(len(neighbors))
                nx, ny = neighbors[idx]
                
                # Carve path
                self.dungeon[ny, nx] = 0
                self.dungeon[cy + (ny - cy) // 2, cx + (nx - cx) // 2] = 0
                
                visited_cells.add((nx, ny))
                stack.append((nx, ny))
            else:
                stack.pop()

        # Find the furthest point from the start for the exit
        max_dist = -1
        furthest_pos = None
        for y in range(1, height, 2):
            for x in range(1, width, 2):
                if self.dungeon[y,x] == 0:
                    dist = self._manhattan_distance(self.start_pos, (x, y))
                    if dist > max_dist:
                        max_dist = dist
                        furthest_pos = (x, y)
        self.exit_pos = furthest_pos
        assert self.exit_pos is not None, "Exit could not be placed"


    def _place_entities(self):
        # Hero
        self.hero = {
            'pos': self.start_pos, 'hp': 20, 'max_hp': 20, 'xp': 0, 'max_xp': 10, 'attack': 5
        }
        
        # Enemies
        self.enemies.clear()
        possible_spawns = []
        for y in range(self.dungeon_height):
            for x in range(self.dungeon_width):
                if self.dungeon[y, x] == 0 and (x, y) != self.start_pos and (x, y) != self.exit_pos:
                    if self._manhattan_distance((x, y), self.start_pos) > 5:
                        possible_spawns.append((x, y))
        
        num_enemies = self._np_random.integers(5, 10)
        if len(possible_spawns) > num_enemies:
            spawn_indices = self._np_random.choice(len(possible_spawns), num_enemies, replace=False)
            
            for i in spawn_indices:
                pos = possible_spawns[i]
                base_hp = 10
                base_attack = 1
                enemy_hp = int(base_hp * (1 + 0.1 * (self.dungeon_level - 1)))
                enemy_attack = max(1, int(base_attack * (1 + 0.1 * (self.dungeon_level - 1))))
                self.enemies.append({'pos': pos, 'hp': enemy_hp, 'max_hp': enemy_hp, 'attack': enemy_attack})
        assert len(self.enemies) > 0, "No enemies were spawned"

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        
        cam_x = self.hero['pos'][0] * self.TILE_SIZE - self.SCREEN_WIDTH // 2 + self.TILE_SIZE // 2
        cam_y = self.hero['pos'][1] * self.TILE_SIZE - self.SCREEN_HEIGHT // 2 + self.TILE_SIZE // 2
        
        self._render_game(cam_x, cam_y)
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self, cam_x, cam_y):
        # Determine visible tile range
        start_col = max(0, cam_x // self.TILE_SIZE)
        end_col = min(self.dungeon_width, (cam_x + self.SCREEN_WIDTH) // self.TILE_SIZE + 2)
        start_row = max(0, cam_y // self.TILE_SIZE)
        end_row = min(self.dungeon_height, (cam_y + self.SCREEN_HEIGHT) // self.TILE_SIZE + 2)

        # Draw dungeon
        for y in range(start_row, end_row):
            for x in range(start_col, end_col):
                screen_x, screen_y = int(x * self.TILE_SIZE - cam_x), int(y * self.TILE_SIZE - cam_y)
                tile_rect = pygame.Rect(screen_x, screen_y, self.TILE_SIZE, self.TILE_SIZE)
                color = self.COLOR_WALL if self.dungeon[y, x] == 1 else self.COLOR_FLOOR
                pygame.draw.rect(self.screen, color, tile_rect)
        
        # Draw exit
        ex, ey = self.exit_pos
        screen_x, screen_y = int(ex * self.TILE_SIZE - cam_x), int(ey * self.TILE_SIZE - cam_y)
        pygame.draw.rect(self.screen, self.COLOR_EXIT, (screen_x, screen_y, self.TILE_SIZE, self.TILE_SIZE))
        pygame.gfxdraw.filled_circle(self.screen, screen_x + self.TILE_SIZE//2, screen_y + self.TILE_SIZE//2, self.TILE_SIZE//3, self.COLOR_BLACK)

        # Draw enemies
        for enemy in self.enemies:
            ex, ey = enemy['pos']
            screen_x, screen_y = int(ex * self.TILE_SIZE - cam_x), int(ey * self.TILE_SIZE - cam_y)
            enemy_rect = pygame.Rect(screen_x, screen_y, self.TILE_SIZE, self.TILE_SIZE)
            pygame.draw.rect(self.screen, self.COLOR_ENEMY, enemy_rect.inflate(-4, -4))
            self._draw_bar(screen_x, screen_y - 8, self.TILE_SIZE, 5, enemy['hp'], enemy['max_hp'], self.COLOR_HEALTH_FG, self.COLOR_HEALTH_BG)

        # Draw hero (always at center)
        hero_screen_pos = (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2)
        hero_rect = pygame.Rect(hero_screen_pos[0] - self.TILE_SIZE//2, hero_screen_pos[1] - self.TILE_SIZE//2, self.TILE_SIZE, self.TILE_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_HERO, hero_rect.inflate(-4, -4))

        # Draw visual feedback
        for feedback in self.action_feedback:
            pos = feedback['pos']
            screen_x = int(pos[0] * self.TILE_SIZE - cam_x)
            screen_y = int(pos[1] * self.TILE_SIZE - cam_y)
            if feedback['type'] == 'hit':
                pygame.draw.rect(self.screen, feedback['color'], (screen_x, screen_y, self.TILE_SIZE, self.TILE_SIZE), 5)
            elif feedback['type'] == 'defend':
                shield_surface = pygame.Surface((self.TILE_SIZE, self.TILE_SIZE), pygame.SRCALPHA)
                pygame.gfxdraw.filled_circle(shield_surface, self.TILE_SIZE//2, self.TILE_SIZE//2, self.TILE_SIZE//2, self.COLOR_DEFEND)
                self.screen.blit(shield_surface, (screen_x, screen_y))
            elif feedback['type'] == 'levelup':
                center_x, center_y = screen_x + self.TILE_SIZE//2, screen_y + self.TILE_SIZE//2
                pygame.gfxdraw.aacircle(self.screen, center_x, center_y, self.TILE_SIZE, self.COLOR_XP)
                pygame.gfxdraw.aacircle(self.screen, center_x, center_y, self.TILE_SIZE-1, self.COLOR_XP)


    def _render_ui(self):
        # Hero stats (above hero sprite)
        hero_screen_x = self.SCREEN_WIDTH // 2 - self.TILE_SIZE // 2
        hero_screen_y = self.SCREEN_HEIGHT // 2 - self.TILE_SIZE // 2
        
        self._draw_bar(hero_screen_x, hero_screen_y - 14, self.TILE_SIZE, 5, self.hero['hp'], self.hero['max_hp'], self.COLOR_HEALTH_FG, self.COLOR_HEALTH_BG)
        self._draw_bar(hero_screen_x, hero_screen_y - 8, self.TILE_SIZE, 5, self.hero['xp'], self.hero['max_xp'], self.COLOR_XP, self.COLOR_BLACK)

        # Top-left UI
        self._draw_text(f"Level: {self.dungeon_level}", (10, 10), self.font_small, self.COLOR_WHITE)
        self._draw_text(f"Score: {self.score}", (10, 30), self.font_small, self.COLOR_WHITE)
        self._draw_text(f"Steps: {self.steps}/{self.MAX_STEPS}", (10, 50), self.font_small, self.COLOR_WHITE)

    def _draw_bar(self, x, y, w, h, val, max_val, fg_color, bg_color):
        if max_val <= 0: return
        ratio = max(0, min(1, val / max_val))
        pygame.draw.rect(self.screen, bg_color, (x, y, w, h))
        pygame.draw.rect(self.screen, fg_color, (x, y, int(w * ratio), h))

    def _draw_text(self, text, pos, font, color):
        text_surface = font.render(text, True, color)
        self.screen.blit(text_surface, pos)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "hero_hp": self.hero['hp'] if self.hero else 0,
            "hero_xp": self.hero['xp'] if self.hero else 0,
            "dungeon_level": self.dungeon_level,
        }

    def _manhattan_distance(self, p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    def validate_implementation(self):
        print("Running implementation validation...")
        # Reset to ensure state is initialized
        _ = self.reset(seed=123)

        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset(seed=456)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc is False
        assert isinstance(info, dict)
        
        # Test specific game assertions
        assert self.hero['hp'] <= self.hero['max_hp'], "Hero HP exceeds max HP"
        for enemy in self.enemies:
            assert enemy['hp'] >= -self.hero['attack'], "Enemy has deeply negative health"
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv()
    obs, info = env.reset()
    
    # Set up Pygame window for human play
    pygame.display.set_caption("Dungeon Crawler")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    terminated = False
    running = True
    
    print("\n" + "="*30)
    print("      MANUAL PLAY TEST")
    print("="*30)
    print(env.user_guide)
    print("Close the window to quit.")
    print("="*30 + "\n")

    last_action_time = pygame.time.get_ticks()
    action_delay = 100 # ms between actions

    while running and not terminated:
        now = pygame.time.get_ticks()
        
        # Default action is no-op
        action = [0, 0, 0]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # Poll for keys and decide on one action
        keys = pygame.key.get_pressed()
        
        # Prioritize actions for human play
        if keys[pygame.K_SPACE]:
            action[1] = 1 # Space
        elif keys[pygame.K_UP]:
            action[0] = 1 # Up
        elif keys[pygame.K_DOWN]:
            action[0] = 2 # Down
        elif keys[pygame.K_LEFT]:
            action[0] = 3 # Left
        elif keys[pygame.K_RIGHT]:
            action[0] = 4 # Right
        
        # Only take an action if enough time has passed
        if (action != [0,0,0]) and (now - last_action_time > action_delay):
            obs, reward, terminated, truncated, info = env.step(action)
            last_action_time = now
            
            if reward != 0:
                print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']}, HP: {info['hero_hp']}")

        # Always render the latest observation
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print("\n--- GAME OVER ---")
            print(f"Final Score: {info['score']}")
            print(f"Reason: {'Victory!' if info['hero_hp'] > 0 and info['steps'] < env.MAX_STEPS else 'Defeat'}")
            pygame.time.wait(3000) # Pause for 3 seconds before closing
            running = False
            
        env.clock.tick(30)

    pygame.quit()