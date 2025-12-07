
# Generated: 2025-08-27T17:18:52.206618
# Source Brief: brief_01492.md
# Brief Index: 1492

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import namedtuple
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Define data structures for game entities
Entity = namedtuple("Entity", ["x", "y", "health"])

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move. Spacebar to attack adjacent enemies. "
        "Each action is one turn."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A turn-based roguelike. Explore dungeons, fight monsters, and collect gold. "
        "Defeat the boss on the third level to win."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    # Screen and Grid
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_WIDTH = 20
    GRID_HEIGHT = 11
    TILE_SIZE = 32
    UI_HEIGHT = SCREEN_HEIGHT - (GRID_HEIGHT * TILE_SIZE) # 48px

    # Colors
    COLOR_BG = (20, 20, 30)
    COLOR_WALL = (70, 70, 90)
    COLOR_FLOOR = (45, 45, 65)
    COLOR_PLAYER = (50, 255, 50)
    COLOR_ENEMY = (255, 50, 50)
    COLOR_BOSS = (200, 50, 255)
    COLOR_GOLD = (255, 223, 0)
    COLOR_STAIRS = (180, 180, 255)
    COLOR_UI_BG = (10, 10, 20)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_HEALTH_BAR_BG = (100, 20, 20)
    COLOR_ATTACK_FX = (255, 255, 255)

    # Game Parameters
    PLAYER_MAX_HEALTH = 100
    ENEMY_MAX_HEALTH = 20
    BOSS_MAX_HEALTH = 60
    PLAYER_ATTACK_DAMAGE = 25
    ENEMY_ATTACK_DAMAGE = 10
    BOSS_ATTACK_DAMAGE = 20
    MAX_STEPS = 1000

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # EXACT spaces:
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 36)

        # Initialize state variables
        self.player = None
        self.enemies = []
        self.boss = None
        self.gold_pieces = []
        self.dungeon_map = None
        self.stairs_pos = None
        self.level = 1
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.last_action_effect = None # For visual feedback

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.level = 1
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player = Entity(x=0, y=0, health=self.PLAYER_MAX_HEALTH)
        self.last_action_effect = None

        self._generate_level()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        self.last_action_effect = None
        reward = -0.01  # Small penalty for each step to encourage efficiency

        # Unpack factorized action
        movement = action[0]
        space_pressed = action[1] == 1
        # shift_pressed is ignored as per the brief's adaptation rules

        # --- Player Action Phase ---
        action_taken = False
        if space_pressed:
            reward += self._handle_attack()
            action_taken = True
        elif movement != 0:
            move_reward, moved = self._handle_movement(movement)
            reward += move_reward
            action_taken = moved
        
        # If no action was taken (e.g. move into wall), it's a "wait" turn.

        # --- Enemy Action Phase ---
        if action_taken:
            reward += self._handle_enemy_turns()

        # --- Check Game State ---
        terminated = False
        if self.player.health <= 0:
            terminated = True
            reward -= 20 # Penalty for dying
        elif self.boss and self.boss.health <= 0:
            terminated = True
            reward += 100 # Big reward for winning
        elif self.steps >= self.MAX_STEPS:
            terminated = True

        self.game_over = terminated

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _generate_level(self):
        self.dungeon_map = np.zeros((self.GRID_WIDTH, self.GRID_HEIGHT), dtype=int)
        self.enemies.clear()
        self.gold_pieces.clear()
        self.boss = None

        # Procedural Generation (BSP-like room placement)
        rooms = []
        max_rooms = self.np_random.integers(5, 8)
        for _ in range(max_rooms):
            w = self.np_random.integers(3, 6)
            h = self.np_random.integers(3, 5)
            x = self.np_random.integers(1, self.GRID_WIDTH - w - 1)
            y = self.np_random.integers(1, self.GRID_HEIGHT - h - 1)
            new_room = pygame.Rect(x, y, w, h)
            
            # Check for overlap
            if not any(new_room.colliderect(r) for r in rooms):
                rooms.append(new_room)

        # Carve rooms and corridors
        for i, room in enumerate(rooms):
            self.dungeon_map[room.left:room.right, room.top:room.bottom] = 1
            if i > 0:
                self._create_corridor(rooms[i-1].center, room.center)

        valid_spawns = list(zip(*np.where(self.dungeon_map == 1)))
        self.np_random.shuffle(valid_spawns)

        # Place Player
        player_pos = valid_spawns.pop()
        self.player = self.player._replace(x=player_pos[0], y=player_pos[1])

        # Place Stairs
        self.stairs_pos = valid_spawns.pop()

        # Place Gold
        num_gold = self.np_random.integers(5, 10)
        for _ in range(min(num_gold, len(valid_spawns))):
            self.gold_pieces.append(valid_spawns.pop())
        
        # Place Enemies/Boss
        if self.level == 3:
            boss_pos = valid_spawns.pop()
            self.boss = Entity(x=boss_pos[0], y=boss_pos[1], health=self.BOSS_MAX_HEALTH)
            num_enemies = 4
        else:
            num_enemies = 2 + self.level * 2

        for _ in range(min(num_enemies, len(valid_spawns))):
            pos = valid_spawns.pop()
            self.enemies.append(Entity(x=pos[0], y=pos[1], health=self.ENEMY_MAX_HEALTH))

    def _create_corridor(self, start_pos, end_pos):
        x1, y1 = start_pos
        x2, y2 = end_pos
        
        if self.np_random.random() > 0.5: # Horizontal then vertical
            self.dungeon_map[min(x1,x2):max(x1,x2)+1, y1] = 1
            self.dungeon_map[x2, min(y1,y2):max(y1,y2)+1] = 1
        else: # Vertical then horizontal
            self.dungeon_map[x1, min(y1,y2):max(y1,y2)+1] = 1
            self.dungeon_map[min(x1,x2):max(x1,x2)+1, y2] = 1

    def _handle_attack(self):
        reward = 0
        attacked = False
        self.last_action_effect = {"type": "attack", "pos": (self.player.x, self.player.y), "radius": 0}
        
        px, py = self.player.x, self.player.y
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            target_x, target_y = px + dx, py + dy
            
            # Check for enemies
            for i, enemy in enumerate(self.enemies):
                if enemy.x == target_x and enemy.y == target_y:
                    new_health = enemy.health - self.PLAYER_ATTACK_DAMAGE
                    self.enemies[i] = enemy._replace(health=new_health)
                    attacked = True
                    if new_health <= 0:
                        reward += 2.0
                        self.score += 20
                    else:
                        reward += 0.5 # Reward for hitting
            
            # Check for boss
            if self.boss and self.boss.x == target_x and self.boss.y == target_y:
                new_health = self.boss.health - self.PLAYER_ATTACK_DAMAGE
                self.boss = self.boss._replace(health=new_health)
                attacked = True
                if new_health <= 0:
                    reward += 10.0
                    self.score += 100
                else:
                    reward += 1.0 # Bigger reward for hitting boss

        # Prune dead enemies
        self.enemies = [e for e in self.enemies if e.health > 0]
        
        if not attacked:
            reward -= 0.1 # Penalty for whiffing
        return reward

    def _handle_movement(self, movement_action):
        reward = 0
        moved = False
        px, py = self.player.x, self.player.y
        
        dist_before = math.hypot(px - self.stairs_pos[0], py - self.stairs_pos[1])

        if movement_action == 1: # Up
            py -= 1
        elif movement_action == 2: # Down
            py += 1
        elif movement_action == 3: # Left
            px -= 1
        elif movement_action == 4: # Right
            px += 1

        if self.dungeon_map[px, py] == 1:
            self.player = self.player._replace(x=px, y=py)
            moved = True
            
            # Reward for moving towards/away from stairs
            dist_after = math.hypot(px - self.stairs_pos[0], py - self.stairs_pos[1])
            if dist_after < dist_before:
                reward += 0.1
            else:
                reward -= 0.1

            # Check for gold
            if (px, py) in self.gold_pieces:
                self.gold_pieces.remove((px, py))
                self.score += 10
                reward += 1.0

            # Check for stairs
            if (px, py) == self.stairs_pos:
                self.level += 1
                self.score += 50
                reward += 5.0
                if self.level <= 3:
                    self._generate_level()
                # Game ends if we go past level 3, but boss logic handles this
        
        return reward, moved

    def _handle_enemy_turns(self):
        reward = 0
        px, py = self.player.x, self.player.y
        
        # Normal Enemies
        for enemy in self.enemies:
            if abs(enemy.x - px) + abs(enemy.y - py) == 1: # Manhattan distance of 1
                self.player = self.player._replace(health=self.player.health - self.ENEMY_ATTACK_DAMAGE)
                reward -= 0.5

        # Boss
        if self.boss and self.boss.health > 0:
            if abs(self.boss.x - px) + abs(self.boss.y - py) == 1:
                self.player = self.player._replace(health=self.player.health - self.BOSS_ATTACK_DAMAGE)
                reward -= 1.0
        
        return reward

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        ts = self.TILE_SIZE
        # Draw dungeon
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                rect = (x * ts, y * ts + self.UI_HEIGHT, ts, ts)
                if self.dungeon_map[x, y] == 1:
                    pygame.draw.rect(self.screen, self.COLOR_FLOOR, rect)
                else:
                    pygame.draw.rect(self.screen, self.COLOR_WALL, rect)
        
        # Draw stairs
        sx, sy = self.stairs_pos
        pygame.draw.rect(self.screen, self.COLOR_STAIRS, (sx * ts, sy * ts + self.UI_HEIGHT, ts, ts))
        
        # Draw gold
        for gx, gy in self.gold_pieces:
            pygame.gfxdraw.filled_circle(self.screen, gx * ts + ts // 2, gy * ts + self.UI_HEIGHT + ts // 2, ts // 4, self.COLOR_GOLD)

        # Draw enemies
        for enemy in self.enemies:
            ex, ey = enemy.x, enemy.y
            pygame.draw.rect(self.screen, self.COLOR_ENEMY, (ex * ts + 4, ey * ts + self.UI_HEIGHT + 4, ts - 8, ts - 8))
        
        # Draw boss
        if self.boss and self.boss.health > 0:
            bx, by = self.boss.x, self.boss.y
            pygame.draw.rect(self.screen, self.COLOR_BOSS, (bx * ts + 2, by * ts + self.UI_HEIGHT + 2, ts - 4, ts - 4))

        # Draw player
        px, py = self.player.x, self.player.y
        player_center_x = px * ts + ts // 2
        player_center_y = py * ts + self.UI_HEIGHT + ts // 2
        pygame.gfxdraw.filled_circle(self.screen, player_center_x, player_center_y, ts // 2 - 2, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, player_center_x, player_center_y, ts // 2 - 2, self.COLOR_PLAYER)
        
        # Draw attack effect
        if self.last_action_effect and self.last_action_effect["type"] == "attack":
            ax, ay = self.last_action_effect["pos"]
            radius = int(self.TILE_SIZE * 0.75)
            pygame.gfxdraw.aacircle(self.screen, ax * ts + ts // 2, ay * ts + self.UI_HEIGHT + ts//2, radius, self.COLOR_ATTACK_FX)
            pygame.gfxdraw.aacircle(self.screen, ax * ts + ts // 2, ay * ts + self.UI_HEIGHT + ts//2, radius-1, self.COLOR_ATTACK_FX)


    def _render_ui(self):
        # Background
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, (0, 0, self.SCREEN_WIDTH, self.UI_HEIGHT))
        
        # Health Bar
        health_pct = max(0, self.player.health / self.PLAYER_MAX_HEALTH)
        hp_bar_width = 200
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (60, 12, hp_bar_width, 24))
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, (60, 12, int(hp_bar_width * health_pct), 24))
        hp_text = self.font_small.render("HP", True, self.COLOR_UI_TEXT)
        self.screen.blit(hp_text, (20, 14))

        # Level
        level_text = self.font_large.render(f"Level: {self.level}", True, self.COLOR_UI_TEXT)
        self.screen.blit(level_text, (self.SCREEN_WIDTH // 2 - level_text.get_width() // 2, 8))
        
        # Gold
        gold_text = self.font_small.render(f"Gold: {self.score}", True, self.COLOR_GOLD)
        self.screen.blit(gold_text, (self.SCREEN_WIDTH - gold_text.get_width() - 20, 14))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "level": self.level,
            "player_health": self.player.health,
            "enemies_left": len(self.enemies) + (1 if self.boss and self.boss.health > 0 else 0)
        }

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
        assert trunc == False
        assert isinstance(info, dict)
        
        # Test game logic assertions
        self.reset()
        self.player = self.player._replace(health=120)
        self._render_ui() # This will clip health to 100 for rendering
        assert self.player.health == 120 # Internal state can exceed
        assert self.dungeon_map[self.player.x, self.player.y] == 1
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup Pygame window for human play
    pygame.display.set_caption("Roguelike Dungeon")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    terminated = False
    
    while not terminated:
        # Default action is "do nothing"
        action = [0, 0, 0] # [move, space, shift]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN:
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
                elif event.key == pygame.K_r: # Reset game
                    obs, info = env.reset()
                    print("--- Game Reset ---")
                
                # If a key was pressed, take a step
                if event.key in [pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT, pygame.K_SPACE]:
                    obs, reward, term, trunc, info = env.step(action)
                    terminated = term
                    print(f"Action: {action}, Reward: {reward:.2f}, Info: {info}")

        # Render the observation to the screen
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit FPS
        
    print("Game Over!")
    env.close()