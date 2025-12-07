
# Generated: 2025-08-27T20:55:09.453046
# Source Brief: brief_02620.md
# Brief Index: 2620

        
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

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move. Space to attack monsters or open chests. "
        "Stand still (no keys) to defend (50% damage reduction)."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Explore a procedurally generated dungeon, battling monsters and "
        "collecting treasure to find the exit."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.TILE_SIZE = 32
        self.MAP_WIDTH, self.MAP_HEIGHT = 50, 50
        self.MAX_STEPS = 1000

        # Colors
        self.COLOR_BG = (10, 10, 20)
        self.COLOR_WALL = (60, 60, 80)
        self.COLOR_FLOOR = (30, 30, 45)
        self.COLOR_PLAYER = (50, 255, 50)
        self.COLOR_PLAYER_GLOW = (150, 255, 150)
        self.COLOR_EXIT = (50, 150, 255)
        self.COLOR_EXIT_GLOW = (150, 200, 255)
        self.COLOR_CHEST = (255, 200, 0)
        self.COLOR_CHEST_GLOW = (255, 230, 100)
        self.COLOR_MONSTER = (255, 50, 50)
        self.COLOR_MONSTER_GLOW = (255, 150, 150)
        self.COLOR_WHITE = (240, 240, 240)
        self.COLOR_RED = (200, 0, 0)
        self.COLOR_GREEN = (0, 200, 0)
        self.COLOR_YELLOW = (200, 200, 0)
        self.COLOR_GREY = (100, 100, 100)
        self.COLOR_UI_BG = (20, 20, 35, 200)

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
        self.font_large = pygame.font.Font(None, 36)
        self.font_damage = pygame.font.Font(None, 20)
        
        # State variables are initialized in reset()
        self.dungeon_map = None
        self.player_pos = None
        self.player_hp = None
        self.player_max_hp = None
        self.player_xp = None
        self.player_xp_needed = None
        self.dungeon_level = None
        self.exit_pos = None
        self.monsters = None
        self.chests = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.last_dist_to_exit = None
        self.animations = None

        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.dungeon_level = 1
        self.player_max_hp = 100
        self.player_hp = self.player_max_hp
        self.player_xp = 0
        self.player_xp_needed = 100
        self.animations = deque()

        self._generate_dungeon()
        self.last_dist_to_exit = self._get_dist_to_exit(self.player_pos)
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        terminated = False
        
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        player_acted = False
        player_defended = False

        # --- Player Turn ---
        target_pos = list(self.player_pos)

        if space_held:
            # Attack
            attacked_monster = False
            for monster in self.monsters:
                if self._is_adjacent(self.player_pos, monster['pos']):
                    damage = self.np_random.integers(15, 26)
                    monster['hp'] -= damage
                    # sfx: player_attack_swoosh.wav
                    self._add_animation('slash', monster['pos'])
                    self._add_animation('damage_text', monster['pos'], text=str(damage), color=self.COLOR_WHITE)
                    if monster['hp'] <= 0:
                        reward += 1.0
                        self.score += 50 * self.dungeon_level
                        self.player_xp += monster['xp_value']
                        # sfx: monster_die.wav
                    attacked_monster = True
                    break # Only attack one monster
            
            # Open Chest
            if not attacked_monster:
                for i, chest_pos in enumerate(self.chests):
                    if tuple(self.player_pos) == chest_pos:
                        xp_gain = 50
                        self.player_xp += xp_gain
                        self.score += xp_gain
                        reward += 0.5
                        self.chests.pop(i)
                        # sfx: chest_open.wav
                        self._add_animation('sparkle', self.player_pos, color=self.COLOR_CHEST)
                        break
            player_acted = True

        elif movement > 0:
            # Move
            if movement == 1: target_pos[1] -= 1 # Up
            elif movement == 2: target_pos[1] += 1 # Down
            elif movement == 3: target_pos[0] -= 1 # Left
            elif movement == 4: target_pos[0] += 1 # Right

            if self.dungeon_map[target_pos[1]][target_pos[0]] == 1: # Is floor
                self.player_pos = target_pos
                player_acted = True
        
        else: # No movement and no space
            player_defended = True
            player_acted = True
            self._add_animation('shield', self.player_pos)
            # sfx: defend_sound.wav
        
        if player_acted:
            # Check for level up
            if self.player_xp >= self.player_xp_needed:
                self.player_xp -= self.player_xp_needed
                self.player_xp_needed = int(self.player_xp_needed * 1.5)
                self.player_max_hp += 20
                self.player_hp = self.player_max_hp # Full heal on level up
                # sfx: level_up.wav
                self._add_animation('sparkle', self.player_pos, color=self.COLOR_GREEN, count=15)

            # --- Monster Turn ---
            for monster in self.monsters:
                if monster['hp'] > 0 and self._is_adjacent(self.player_pos, monster['pos']):
                    damage = monster['attack']
                    if player_defended:
                        damage = max(1, int(damage / 2))
                    self.player_hp -= damage
                    # sfx: player_hit.wav
                    self._add_animation('slash', self.player_pos, color=self.COLOR_MONSTER)
                    self._add_animation('damage_text', self.player_pos, text=str(damage), color=self.COLOR_RED)

            # Cleanup dead monsters
            self.monsters = [m for m in self.monsters if m['hp'] > 0]
            
            # Distance reward
            new_dist_to_exit = self._get_dist_to_exit(self.player_pos)
            if new_dist_to_exit < self.last_dist_to_exit:
                reward += 0.1
            elif new_dist_to_exit > self.last_dist_to_exit:
                reward -= 0.1
            self.last_dist_to_exit = new_dist_to_exit

        # --- Check Game End Conditions ---
        if self.player_hp <= 0:
            reward = -100
            terminated = True
            self.game_over = True
            # sfx: game_over.wav
        
        if tuple(self.player_pos) == self.exit_pos:
            reward = 100
            self.score += 1000 * self.dungeon_level
            terminated = True
            self.game_over = True
            # sfx: victory_fanfare.wav

        self.steps += 1
        if self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True

        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

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
            "hp": self.player_hp,
            "level": self.dungeon_level,
            "xp": self.player_xp,
        }

    def _generate_dungeon(self):
        self.dungeon_map = np.zeros((self.MAP_HEIGHT, self.MAP_WIDTH), dtype=int) # 0=wall, 1=floor
        
        # Randomized DFS for maze generation
        stack = []
        start_x, start_y = self.np_random.integers(1, self.MAP_WIDTH-1, 2)
        if start_x % 2 == 0: start_x += 1
        if start_y % 2 == 0: start_y += 1
        
        self.dungeon_map[start_y][start_x] = 1
        stack.append((start_x, start_y))
        
        floor_tiles = []

        while stack:
            x, y = stack[-1]
            floor_tiles.append((x,y))
            neighbors = []
            for dx, dy in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
                nx, ny = x + dx, y + dy
                if 0 < nx < self.MAP_WIDTH-1 and 0 < ny < self.MAP_HEIGHT-1 and self.dungeon_map[ny][nx] == 0:
                    neighbors.append((nx, ny))
            
            if neighbors:
                nx, ny = self.np_random.choice(neighbors, axis=0)
                self.dungeon_map[ny][nx] = 1
                self.dungeon_map[y + (ny-y)//2][x + (nx-x)//2] = 1
                stack.append((nx, ny))
            else:
                stack.pop()

        # Place player, exit, monsters, chests
        self.np_random.shuffle(floor_tiles)
        self.player_pos = list(floor_tiles.pop(0))
        self.exit_pos = floor_tiles.pop(0)
        
        num_monsters = self.np_random.integers(5, 10 + self.dungeon_level)
        num_chests = self.np_random.integers(3, 6)
        
        self.monsters = []
        for _ in range(num_monsters):
            if not floor_tiles: break
            pos = floor_tiles.pop(0)
            base_hp = 20
            base_attack = 5
            scale = 1 + (self.dungeon_level - 1) * 0.1
            self.monsters.append({
                'pos': list(pos),
                'hp': int(base_hp * scale),
                'max_hp': int(base_hp * scale),
                'attack': int(base_attack * scale),
                'xp_value': int(10 * scale)
            })

        self.chests = []
        for _ in range(num_chests):
            if not floor_tiles: break
            self.chests.append(floor_tiles.pop(0))

    def _render_game(self):
        cam_x = self.player_pos[0] * self.TILE_SIZE - self.WIDTH // 2
        cam_y = self.player_pos[1] * self.TILE_SIZE - self.HEIGHT // 2

        start_col = max(0, cam_x // self.TILE_SIZE)
        end_col = min(self.MAP_WIDTH, (cam_x + self.WIDTH) // self.TILE_SIZE + 2)
        start_row = max(0, cam_y // self.TILE_SIZE)
        end_row = min(self.MAP_HEIGHT, (cam_y + self.HEIGHT) // self.TILE_SIZE + 2)

        for y in range(start_row, end_row):
            for x in range(start_col, end_col):
                screen_x, screen_y = x * self.TILE_SIZE - cam_x, y * self.TILE_SIZE - cam_y
                color = self.COLOR_FLOOR if self.dungeon_map[y][x] == 1 else self.COLOR_WALL
                pygame.draw.rect(self.screen, color, (screen_x, screen_y, self.TILE_SIZE, self.TILE_SIZE))

        # Draw exit
        screen_x, screen_y = self.exit_pos[0] * self.TILE_SIZE - cam_x, self.exit_pos[1] * self.TILE_SIZE - cam_y
        pygame.gfxdraw.filled_circle(self.screen, screen_x + self.TILE_SIZE//2, screen_y + self.TILE_SIZE//2, self.TILE_SIZE//2, self.COLOR_EXIT_GLOW)
        pygame.gfxdraw.filled_circle(self.screen, screen_x + self.TILE_SIZE//2, screen_y + self.TILE_SIZE//2, self.TILE_SIZE//2 - 4, self.COLOR_EXIT)

        # Draw chests
        for x, y in self.chests:
            screen_x, screen_y = x * self.TILE_SIZE - cam_x, y * self.TILE_SIZE - cam_y
            pygame.draw.rect(self.screen, self.COLOR_CHEST_GLOW, (screen_x+2, screen_y+2, self.TILE_SIZE-4, self.TILE_SIZE-4))
            pygame.draw.rect(self.screen, self.COLOR_CHEST, (screen_x+4, screen_y+4, self.TILE_SIZE-8, self.TILE_SIZE-8))

        # Draw monsters
        for monster in self.monsters:
            x, y = monster['pos']
            screen_x, screen_y = x * self.TILE_SIZE - cam_x, y * self.TILE_SIZE - cam_y
            pygame.draw.rect(self.screen, self.COLOR_MONSTER_GLOW, (screen_x+2, screen_y+2, self.TILE_SIZE-4, self.TILE_SIZE-4))
            pygame.draw.rect(self.screen, self.COLOR_MONSTER, (screen_x+4, screen_y+4, self.TILE_SIZE-8, self.TILE_SIZE-8))
            # Health bar
            hp_ratio = monster['hp'] / monster['max_hp']
            pygame.draw.rect(self.screen, self.COLOR_RED, (screen_x+4, screen_y-8, self.TILE_SIZE-8, 5))
            pygame.draw.rect(self.screen, self.COLOR_GREEN, (screen_x+4, screen_y-8, (self.TILE_SIZE-8)*hp_ratio, 5))

        # Draw player
        screen_x, screen_y = self.player_pos[0] * self.TILE_SIZE - cam_x, self.player_pos[1] * self.TILE_SIZE - cam_y
        pygame.draw.rect(self.screen, self.COLOR_PLAYER_GLOW, (screen_x+2, screen_y+2, self.TILE_SIZE-4, self.TILE_SIZE-4))
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, (screen_x+4, screen_y+4, self.TILE_SIZE-8, self.TILE_SIZE-8))
        
        # Draw animations
        new_animations = deque()
        while self.animations:
            anim = self.animations.popleft()
            anim['timer'] -= 1
            x, y = anim['pos']
            screen_x, screen_y = x * self.TILE_SIZE - cam_x, y * self.TILE_SIZE - cam_y
            
            if anim['type'] == 'slash':
                color = anim.get('color', self.COLOR_WHITE)
                pygame.draw.line(self.screen, color, (screen_x+4, screen_y+4), (screen_x+self.TILE_SIZE-4, screen_y+self.TILE_SIZE-4), 3)
                pygame.draw.line(self.screen, color, (screen_x+4, screen_y+self.TILE_SIZE-4), (screen_x+self.TILE_SIZE-4, screen_y+4), 3)
            elif anim['type'] == 'shield':
                pygame.gfxdraw.arc(self.screen, screen_x+self.TILE_SIZE//2, screen_y+self.TILE_SIZE//2, self.TILE_SIZE//2, 90, 270, self.COLOR_WHITE)
            elif anim['type'] == 'sparkle':
                for _ in range(anim.get('count', 5)):
                    px = screen_x + self.np_random.integers(0, self.TILE_SIZE)
                    py = screen_y + self.np_random.integers(0, self.TILE_SIZE)
                    pygame.draw.circle(self.screen, anim['color'], (px, py), self.np_random.integers(1, 4))
            elif anim['type'] == 'damage_text':
                text_surf = self.font_damage.render(anim['text'], True, anim['color'])
                self.screen.blit(text_surf, (screen_x + 8, screen_y - 15 + anim['timer']))

            if anim['timer'] > 0:
                new_animations.append(anim)
        self.animations = new_animations

    def _render_ui(self):
        # UI Panel
        ui_panel = pygame.Surface((self.WIDTH, 80), pygame.SRCALPHA)
        ui_panel.fill(self.COLOR_UI_BG)
        self.screen.blit(ui_panel, (0, 0))
        
        # HP Bar
        self._draw_text("HP", 20, 15, self.font_small)
        hp_ratio = max(0, self.player_hp / self.player_max_hp)
        pygame.draw.rect(self.screen, self.COLOR_RED, (60, 10, 200, 20))
        pygame.draw.rect(self.screen, self.COLOR_GREEN, (60, 10, 200 * hp_ratio, 20))
        hp_text = f"{self.player_hp}/{self.player_max_hp}"
        self._draw_text(hp_text, 160, 15, self.font_small, center=True)

        # XP Bar
        self._draw_text("XP", 20, 45, self.font_small)
        xp_ratio = self.player_xp / self.player_xp_needed
        pygame.draw.rect(self.screen, self.COLOR_GREY, (60, 40, 200, 20))
        pygame.draw.rect(self.screen, self.COLOR_YELLOW, (60, 40, 200 * xp_ratio, 20))
        xp_text = f"{self.player_xp}/{self.player_xp_needed}"
        self._draw_text(xp_text, 160, 45, self.font_small, center=True)
        
        # Level and Score
        self._draw_text(f"Level: {self.dungeon_level}", 300, 15, self.font_small)
        self._draw_text(f"Score: {int(self.score)}", 300, 45, self.font_small)
        self._draw_text(f"Steps: {self.steps}/{self.MAX_STEPS}", 450, 15, self.font_small)

        if self.game_over:
            s = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            s.fill((0,0,0,128))
            self.screen.blit(s, (0,0))
            msg = "VICTORY!" if self.player_hp > 0 else "GAME OVER"
            color = self.COLOR_GREEN if self.player_hp > 0 else self.COLOR_RED
            self._draw_text(msg, self.WIDTH // 2, self.HEIGHT // 2 - 20, self.font_large, color, center=True)

    def _draw_text(self, text, x, y, font, color=None, center=False):
        if color is None: color = self.COLOR_WHITE
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect()
        if center:
            text_rect.center = (x, y)
        else:
            text_rect.topleft = (x, y)
        self.screen.blit(text_surface, text_rect)

    def _get_dist_to_exit(self, pos):
        return abs(pos[0] - self.exit_pos[0]) + abs(pos[1] - self.exit_pos[1])

    def _is_adjacent(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1]) == 1

    def _add_animation(self, anim_type, pos, **kwargs):
        anim = {'type': anim_type, 'pos': pos, 'timer': 5}
        anim.update(kwargs)
        self.animations.append(anim)

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert obs.dtype == np.uint8
        assert isinstance(info, dict)
        
        # Test observation space after reset
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Dungeon Crawler")
    clock = pygame.time.Clock()
    
    terminated = False
    
    while not terminated:
        movement = 0 # No-op
        space_held = 0
        shift_held = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1

        action = [movement, space_held, shift_held]
        
        # In manual play, we only step on key press to simulate turns
        if any(keys):
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Action: {action}, Reward: {reward:.2f}, Info: {info}")

        # Render the observation from the environment
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(10) # Limit speed of manual turns

    pygame.quit()