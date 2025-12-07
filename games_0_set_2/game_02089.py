
# Generated: 2025-08-28T03:39:34.502191
# Source Brief: brief_02089.md
# Brief Index: 2089

        
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
        "Controls: Arrow keys to move. Space to attack in the direction you are facing. "
        "Collect gold and reach the exit."
    )

    game_description = (
        "A turn-based dungeon crawler. Navigate a procedural dungeon, battle enemies, and collect "
        "at least 50 gold to escape through the exit."
    )

    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    TILE_SIZE = 32
    MAX_STEPS = 1000
    WIN_GOLD_REQ = 50

    # Colors
    COLOR_BG = (10, 10, 20)
    COLOR_WALL = (60, 60, 80)
    COLOR_FLOOR = (30, 30, 45)
    COLOR_HERO = (50, 150, 255)
    COLOR_EXIT = (255, 215, 0)
    COLOR_GOLD = (255, 223, 0)
    COLOR_TEXT = (240, 240, 240)
    COLOR_HEALTH_BG = (100, 0, 0)
    COLOR_HEALTH_FG = (0, 200, 0)
    ENEMY_COLORS = [
        (255, 80, 80),   # 0: Chaser (Red)
        (80, 255, 80),   # 1: Patrol (Green)
        (80, 80, 255),   # 2: Chaser (Blue)
        (200, 80, 200),  # 3: Coward (Purple)
        (255, 165, 0),   # 4: Stationary (Orange)
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 36)
        
        # Game state variables are initialized in reset()
        self.dungeon_layout = None
        self.hero_pos = None
        self.hero_health = None
        self.hero_max_health = None
        self.hero_facing = None
        self.exit_pos = None
        self.enemies = None
        self.gold_piles = None
        self.gold_count = None
        self.steps = None
        self.score = None
        self.np_random = None
        self.particles = []

        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        
        self.hero_health = 100
        self.hero_max_health = 100
        self.hero_facing = (0, 1)  # Start facing down
        self.gold_count = 0
        self.particles = []

        self._generate_dungeon(width=50, height=50)
        self._populate_dungeon()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, _ = action
        space_pressed = space_held == 1

        reward = 0
        terminated = False
        
        # --- Player Turn ---
        # Prioritize attack
        if space_pressed:
            reward += self._handle_player_attack()
        # Then movement
        elif movement > 0:
            reward += self._handle_player_move(movement)
        
        # If no action (movement=0, space=0), it's a "wait" turn.
        # This still triggers enemy turns.

        # --- Enemy Turn ---
        if self.hero_health > 0:
            reward += self._handle_enemy_turns()

        # Check for game over from enemy attacks
        if self.hero_health <= 0:
            terminated = True
            reward = -100.0

        # Check for termination from step limit
        self.steps += 1
        if self.steps >= self.MAX_STEPS:
            terminated = True

        # Check for win condition (must be done after move)
        if self.hero_pos == self.exit_pos and self.gold_count >= self.WIN_GOLD_REQ:
            terminated = True
            reward = 100.0
        
        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _generate_dungeon(self, width, height):
        self.dungeon_layout = np.ones((width, height), dtype=int) # 1 = wall
        
        # Random walk to carve out floor
        start_x, start_y = width // 2, height // 2
        px, py = start_x, start_y
        self.dungeon_layout[px, py] = 0 # 0 = floor
        
        num_steps = (width * height) // 2
        for _ in range(num_steps):
            # Choose a random direction
            dx, dy = self.np_random.choice([(-1, 0), (1, 0), (0, -1), (0, 1)], p=[0.25, 0.25, 0.25, 0.25])
            # Move two steps to create wider corridors
            for _ in range(2):
                px = np.clip(px + dx, 1, width - 2)
                py = np.clip(py + dy, 1, height - 2)
                self.dungeon_layout[px, py] = 0

        self.hero_pos = (start_x, start_y)

    def _populate_dungeon(self):
        floor_tiles = np.argwhere(self.dungeon_layout == 0).tolist()
        self.np_random.shuffle(floor_tiles)

        # Remove hero start position from available tiles
        if list(self.hero_pos) in floor_tiles:
            floor_tiles.remove(list(self.hero_pos))

        # Place Exit
        self.exit_pos = tuple(floor_tiles.pop())
        
        # Place Gold
        num_gold = 20
        self.gold_piles = [tuple(floor_tiles.pop()) for _ in range(num_gold) if floor_tiles]

        # Place Enemies
        num_enemies = 15
        self.enemies = []
        enemy_max_health = 20 + (self.gold_count // 10)
        for i in range(num_enemies):
            if not floor_tiles: break
            enemy_pos = tuple(floor_tiles.pop())
            enemy_type = self.np_random.integers(0, len(self.ENEMY_COLORS))
            self.enemies.append({
                'pos': enemy_pos,
                'health': enemy_max_health,
                'max_health': enemy_max_health,
                'type': enemy_type,
                'id': i
            })

    def _handle_player_attack(self):
        # sound: player_attack.wav
        target_pos = (self.hero_pos[0] + self.hero_facing[0], self.hero_pos[1] + self.hero_facing[1])
        reward = 0
        
        self._create_particles(target_pos, self.COLOR_HERO, 5, 0.2) # Attack visual effect

        for enemy in self.enemies[:]:
            if enemy['pos'] == target_pos:
                enemy['health'] -= 10
                if enemy['health'] <= 0:
                    self.enemies.remove(enemy)
                    reward += 5.0 # Defeated enemy reward
                    # sound: enemy_die.wav
                else:
                    # sound: enemy_hit.wav
                    pass
                break # Only attack one enemy
        return reward
    
    def _handle_player_move(self, movement):
        reward = 0
        old_pos = self.hero_pos
        dx, dy = 0, 0
        if movement == 1: dx, dy = 0, -1  # Up
        elif movement == 2: dx, dy = 0, 1  # Down
        elif movement == 3: dx, dy = -1, 0 # Left
        elif movement == 4: dx, dy = 1, 0  # Right
        
        self.hero_facing = (dx, dy)
        new_pos = (old_pos[0] + dx, old_pos[1] + dy)
        
        # Check for wall collision
        if self.dungeon_layout[new_pos[0], new_pos[1]] == 1:
            # sound: bump_wall.wav
            return 0 # No move, no reward change
        
        # Check for enemy collision
        if any(enemy['pos'] == new_pos for enemy in self.enemies):
            return 0 # Can't move into an enemy's space

        # Valid move
        # sound: player_step.wav
        dist_before = abs(old_pos[0] - self.exit_pos[0]) + abs(old_pos[1] - self.exit_pos[1])
        dist_after = abs(new_pos[0] - self.exit_pos[0]) + abs(new_pos[1] - self.exit_pos[1])
        
        if dist_after < dist_before:
            reward += 0.1
        elif dist_after > dist_before:
            reward -= 0.1
            
        self.hero_pos = new_pos
        
        # Check for gold collection
        if self.hero_pos in self.gold_piles:
            self.gold_piles.remove(self.hero_pos)
            self.gold_count += 1
            reward += 1.0
            # sound: collect_gold.wav
            self._create_particles(self.hero_pos, self.COLOR_GOLD, 10, 0.3)

        return reward

    def _handle_enemy_turns(self):
        reward = 0
        hero_attacked = False
        
        occupied_tiles = {e['pos'] for e in self.enemies}
        occupied_tiles.add(self.hero_pos)

        for enemy in self.enemies:
            old_pos = enemy['pos']
            ex, ey = old_pos
            hx, hy = self.hero_pos
            dist_to_hero = math.hypot(ex - hx, ey - hy)

            # --- AI Logic ---
            dx, dy = 0, 0
            if enemy['type'] == 0 or enemy['type'] == 2: # Chaser
                if dist_to_hero < 8:
                    if hx > ex: dx = 1
                    elif hx < ex: dx = -1
                    if hy > ey: dy = 1
                    elif hy < ey: dy = -1
            elif enemy['type'] == 1: # Patrol (simple random walk)
                if self.np_random.random() < 0.2: # 20% chance to change direction
                    dx, dy = self.np_random.choice([(-1,0), (1,0), (0,-1), (0,1)], p=[0.25, 0.25, 0.25, 0.25])
                else: # Continue in current direction if possible
                    dx, dy = enemy.get('dir', (1,0))
                enemy['dir'] = (dx, dy)
            elif enemy['type'] == 3: # Coward
                if dist_to_hero < 5:
                    if hx > ex: dx = -1
                    elif hx < ex: dx = 1
                    if hy > ey: dy = -1
                    elif hy < ey: dy = 1
            # Type 4 (Stationary) does nothing.

            # --- Move Resolution ---
            if dx != 0 or dy != 0:
                # Try moving along one axis at a time
                if dx != 0 and dy != 0:
                    if self.np_random.random() < 0.5: dy = 0
                    else: dx = 0
                
                new_pos = (ex + dx, ey + dy)

                if new_pos == self.hero_pos:
                    # Attack hero instead of moving
                    self.hero_health = max(0, self.hero_health - 5)
                    hero_attacked = True
                    # sound: hero_hit.wav
                elif self.dungeon_layout[new_pos[0], new_pos[1]] == 0 and new_pos not in occupied_tiles:
                    occupied_tiles.remove(old_pos)
                    enemy['pos'] = new_pos
                    occupied_tiles.add(new_pos)

        if hero_attacked:
            self._create_particles(self.hero_pos, (255, 0, 0), 10, 0.3)

        return reward

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Camera centered on hero
        cam_x = self.hero_pos[0] * self.TILE_SIZE - self.SCREEN_WIDTH / 2
        cam_y = self.hero_pos[1] * self.TILE_SIZE - self.SCREEN_HEIGHT / 2

        # Get visible tile range
        start_x = max(0, int(cam_x / self.TILE_SIZE))
        end_x = min(self.dungeon_layout.shape[0], int((cam_x + self.SCREEN_WIDTH) / self.TILE_SIZE) + 1)
        start_y = max(0, int(cam_y / self.TILE_SIZE))
        end_y = min(self.dungeon_layout.shape[1], int((cam_y + self.SCREEN_HEIGHT) / self.TILE_SIZE) + 1)

        # Render dungeon
        for x in range(start_x, end_x):
            for y in range(start_y, end_y):
                screen_x, screen_y = x * self.TILE_SIZE - cam_x, y * self.TILE_SIZE - cam_y
                color = self.COLOR_WALL if self.dungeon_layout[x, y] == 1 else self.COLOR_FLOOR
                pygame.draw.rect(self.screen, color, (screen_x, screen_y, self.TILE_SIZE, self.TILE_SIZE))

        # Render exit
        exit_screen_x = self.exit_pos[0] * self.TILE_SIZE - cam_x
        exit_screen_y = self.exit_pos[1] * self.TILE_SIZE - cam_y
        pygame.draw.rect(self.screen, self.COLOR_EXIT, (exit_screen_x, exit_screen_y, self.TILE_SIZE, self.TILE_SIZE))
        pygame.draw.rect(self.screen, (0,0,0), (exit_screen_x+4, exit_screen_y+4, self.TILE_SIZE-8, self.TILE_SIZE-8), 2)


        # Render gold
        for gx, gy in self.gold_piles:
            gold_screen_x = gx * self.TILE_SIZE - cam_x + self.TILE_SIZE // 2
            gold_screen_y = gy * self.TILE_SIZE - cam_y + self.TILE_SIZE // 2
            pygame.draw.circle(self.screen, self.COLOR_GOLD, (gold_screen_x, gold_screen_y), self.TILE_SIZE // 4)

        # Render particles
        self._update_and_draw_particles(cam_x, cam_y)

        # Render enemies
        for enemy in self.enemies:
            ex, ey = enemy['pos']
            enemy_screen_x = ex * self.TILE_SIZE - cam_x
            enemy_screen_y = ey * self.TILE_SIZE - cam_y
            color = self.ENEMY_COLORS[enemy['type']]
            pygame.draw.rect(self.screen, color, (enemy_screen_x + 4, enemy_screen_y + 4, self.TILE_SIZE - 8, self.TILE_SIZE - 8))
            self._draw_health_bar(enemy_screen_x, enemy_screen_y - 8, self.TILE_SIZE, 5, enemy['health'], enemy['max_health'])
            
        # Render hero
        hx, hy = self.hero_pos
        hero_screen_x = hx * self.TILE_SIZE - cam_x
        hero_screen_y = hy * self.TILE_SIZE - cam_y
        pygame.draw.rect(self.screen, self.COLOR_HERO, (hero_screen_x + 2, hero_screen_y + 2, self.TILE_SIZE - 4, self.TILE_SIZE - 4))
        # Facing indicator
        facing_x = hero_screen_x + self.TILE_SIZE/2 + self.hero_facing[0] * self.TILE_SIZE/3
        facing_y = hero_screen_y + self.TILE_SIZE/2 + self.hero_facing[1] * self.TILE_SIZE/3
        pygame.draw.circle(self.screen, (255,255,255), (facing_x, facing_y), 3)

    def _render_ui(self):
        # Gold count
        gold_text = f"Gold: {self.gold_count}"
        self._draw_text(gold_text, (10, 10), self.font_large, self.COLOR_GOLD)

        # Hero health
        health_text = f"Health: {self.hero_health}"
        text_surf = self.font_large.render(health_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surf, (self.SCREEN_WIDTH - text_surf.get_width() - 10, 10))
        self._draw_health_bar(
            self.SCREEN_WIDTH - 160, 40, 150, 15, self.hero_health, self.hero_max_health
        )

        # Steps
        steps_text = f"Turns: {self.steps}/{self.MAX_STEPS}"
        self._draw_text(steps_text, (10, 40), self.font_small, self.COLOR_TEXT)

    def _draw_text(self, text, pos, font, color):
        text_surface = font.render(text, True, color)
        self.screen.blit(text_surface, pos)

    def _draw_health_bar(self, x, y, w, h, current, maximum):
        if maximum <= 0: return
        ratio = max(0, current / maximum)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BG, (x, y, w, h))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_FG, (x, y, int(w * ratio), h))

    def _create_particles(self, pos_world, color, count, lifetime):
        for _ in range(count):
            self.particles.append({
                'pos_world': list(pos_world),
                'vel': [self.np_random.uniform(-1, 1) * 0.5, self.np_random.uniform(-1, 1) * 0.5],
                'lifetime': lifetime,
                'max_lifetime': lifetime,
                'color': color
            })

    def _update_and_draw_particles(self, cam_x, cam_y):
        for p in self.particles[:]:
            p['pos_world'][0] += p['vel'][0]
            p['pos_world'][1] += p['vel'][1]
            p['lifetime'] -= 1/30.0 # Assume 30 FPS for lifetime decay

            if p['lifetime'] <= 0:
                self.particles.remove(p)
                continue

            alpha = int(255 * (p['lifetime'] / p['max_lifetime']))
            color = (*p['color'], alpha)
            
            screen_x = p['pos_world'][0] * self.TILE_SIZE - cam_x + self.TILE_SIZE // 2
            screen_y = p['pos_world'][1] * self.TILE_SIZE - cam_y + self.TILE_SIZE // 2
            
            size = int(5 * (p['lifetime'] / p['max_lifetime']))
            if size > 0:
                s = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
                pygame.draw.circle(s, color, (size, size), size)
                self.screen.blit(s, (int(screen_x - size), int(screen_y - size)))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "gold": self.gold_count,
            "health": self.hero_health,
            "pos": self.hero_pos,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert obs.dtype == np.uint8
        assert isinstance(info, dict)
        
        # Test observation space from a direct call
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
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
    # This block allows you to play the game manually
    env = GameEnv()
    obs, info = env.reset()
    
    # Pygame setup for manual play
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Dungeon Crawler")
    clock = pygame.time.Clock()

    action = [0, 0, 0] # no-op, no-fire, no-shift
    running = True
    terminated = False

    while running:
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r: # Reset game
                    obs, info = env.reset()
                    terminated = False
                    print("--- Game Reset ---")
                if terminated: continue

                # Map keys to actions
                movement = 0
                space = 0
                if event.key == pygame.K_UP: movement = 1
                elif event.key == pygame.K_DOWN: movement = 2
                elif event.key == pygame.K_LEFT: movement = 3
                elif event.key == pygame.K_RIGHT: movement = 4
                elif event.key == pygame.K_SPACE: space = 1
                elif event.key == pygame.K_w: movement = 0 # Wait
                
                # Only step if a valid action key is pressed
                if movement > 0 or space > 0 or event.key == pygame.K_w:
                    action = [movement, space, 0]
                    obs, reward, terminated, truncated, info = env.step(action)
                    print(f"Action: {action}, Reward: {reward:.2f}, Score: {info['score']:.2f}, Gold: {info['gold']}, Terminated: {terminated}")

        # Drawing
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        if terminated:
            # Display game over message
            s = pygame.Surface((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT), pygame.SRCALPHA)
            s.fill((0,0,0,180))
            screen.blit(s, (0,0))
            
            win_text = "YOU ESCAPED!" if info['gold'] >= GameEnv.WIN_GOLD_REQ and info['health'] > 0 else "YOU DIED"
            font = pygame.font.Font(None, 72)
            text = font.render(win_text, True, (255, 255, 255))
            text_rect = text.get_rect(center=(GameEnv.SCREEN_WIDTH/2, GameEnv.SCREEN_HEIGHT/2 - 20))
            screen.blit(text, text_rect)

            font_small = pygame.font.Font(None, 36)
            sub_text = font_small.render("Press 'R' to restart", True, (200, 200, 200))
            sub_text_rect = sub_text.get_rect(center=(GameEnv.SCREEN_WIDTH/2, GameEnv.SCREEN_HEIGHT/2 + 30))
            screen.blit(sub_text, sub_text_rect)

        pygame.display.flip()
        clock.tick(30) # Limit frame rate

    env.close()