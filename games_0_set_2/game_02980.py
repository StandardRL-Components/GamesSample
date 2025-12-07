
# Generated: 2025-08-27T22:00:31.363433
# Source Brief: brief_02980.md
# Brief Index: 2980

        
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
    """
    A grid-based, rogue-lite dungeon crawler where the player navigates a
    procedurally generated level, battles enemies, and collects gold to
    reach the exit. The game is turn-based and features pixel-art-style
    visuals with particle effects for enhanced game feel.
    """
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrow keys to move. Hold Space to attack in your last-moved direction. "
        "Hold Shift to use a collected health potion."
    )

    game_description = (
        "Navigate a dangerous dungeon, battling monsters and collecting treasure. "
        "Reach the blue exit tile to win. Your health is limited, but potions can be found."
    )

    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_WIDTH, GRID_HEIGHT = 20, 12
    TILE_SIZE = 32
    
    MAX_STEPS = 1000
    INITIAL_ENEMIES = 2
    INITIAL_POTIONS_ON_MAP = 3
    INITIAL_GOLD_PILES = 5
    
    # --- Colors ---
    COLOR_BG = (20, 20, 30)
    COLOR_FLOOR = (40, 40, 60)
    COLOR_WALL = (80, 80, 100)
    COLOR_HERO = (0, 255, 255)
    COLOR_ENEMY = (255, 60, 60)
    COLOR_GOLD = (255, 220, 0)
    COLOR_POTION = (60, 255, 60)
    COLOR_EXIT = (60, 60, 255)
    COLOR_TEXT = (240, 240, 240)
    COLOR_HEALTH_FG = (60, 200, 60)
    COLOR_HEALTH_BG = (120, 40, 40)

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
        self.font_ui = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_msg = pygame.font.SysFont("Consolas", 32, bold=True)
        
        # This will be initialized in reset()
        self.rng = None
        
        # Game state variables
        self.grid = None
        self.hero_pos = None
        self.hero_health = None
        self.hero_max_health = 3
        self.potions_held = 0
        self.last_move_dir = (1, 0) # Start facing right
        self.enemies = []
        self.gold = []
        self.potions_on_map = []
        self.exit_pos = None
        
        # Episode variables
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        
        # Visuals
        self.particles = []
        self.world_offset = [0, 0]

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.rng = np.random.default_rng(seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        self.hero_health = self.hero_max_health
        self.potions_held = 0
        self.particles = []

        self._generate_level()

        return self._get_observation(), self._get_info()

    def _generate_level(self):
        self.grid = np.zeros((self.GRID_WIDTH, self.GRID_HEIGHT), dtype=int)
        
        # Simple obstacle generation
        for _ in range(int(self.GRID_WIDTH * self.GRID_HEIGHT * 0.2)):
            x, y = self.rng.integers(0, self.GRID_WIDTH), self.rng.integers(0, self.GRID_HEIGHT)
            self.grid[x, y] = 1 # Wall
        
        # Define start and end points
        self.hero_pos = np.array([1, self.GRID_HEIGHT // 2])
        self.exit_pos = np.array([self.GRID_WIDTH - 2, self.GRID_HEIGHT // 2])
        
        # Ensure start and end are clear
        self.grid[self.hero_pos[0], self.hero_pos[1]] = 0
        self.grid[self.exit_pos[0], self.exit_pos[1]] = 0

        # Ensure a path exists using flood fill
        q = deque([tuple(self.hero_pos)])
        reachable = {tuple(self.hero_pos)}
        while q:
            x, y = q.popleft()
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT and self.grid[nx, ny] == 0 and (nx, ny) not in reachable:
                    reachable.add((nx, ny))
                    q.append((nx, ny))
        
        # If exit is not reachable, clear a path
        if tuple(self.exit_pos) not in reachable:
            for x in range(self.GRID_WIDTH):
                for y in range(self.GRID_HEIGHT):
                    if (x, y) not in reachable:
                        self.grid[x, y] = 0 # Clear all unreachable cells to guarantee a path
            # Re-run flood fill to get the new full set of reachable tiles
            q = deque([tuple(self.hero_pos)])
            reachable = {tuple(self.hero_pos)}
            while q:
                x, y = q.popleft()
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT and self.grid[nx, ny] == 0 and (nx, ny) not in reachable:
                        reachable.add((nx, ny))
                        q.append((nx, ny))

        # Populate level
        spawn_locations = list(reachable - {tuple(self.hero_pos), tuple(self.exit_pos)})
        self.rng.shuffle(spawn_locations)
        
        num_enemies = self.INITIAL_ENEMIES
        self.enemies = [{'pos': np.array(pos)} for pos in spawn_locations[:num_enemies]]
        
        num_gold = self.INITIAL_GOLD_PILES
        self.gold = [np.array(pos) for pos in spawn_locations[num_enemies:num_enemies+num_gold]]
        
        num_potions = self.INITIAL_POTIONS_ON_MAP
        self.potions_on_map = [np.array(pos) for pos in spawn_locations[num_enemies+num_gold:num_enemies+num_gold+num_potions]]


    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0.0
        terminated = False
        
        old_dist_to_exit = np.linalg.norm(self.hero_pos - self.exit_pos)
        
        # --- Player Turn ---
        action_taken = False
        
        # Priority 1: Attack
        if space_held and not action_taken:
            action_taken = True
            target_pos = self.hero_pos + self.last_move_dir
            enemy_hit = None
            for enemy in self.enemies:
                if np.array_equal(enemy['pos'], target_pos):
                    enemy_hit = enemy
                    break
            if enemy_hit:
                # Sound: Player attack, enemy hit
                self.enemies.remove(enemy_hit)
                reward += 1.0
                self.score += 50
                self._add_particles(target_pos * self.TILE_SIZE + self.TILE_SIZE / 2, 20, self.COLOR_ENEMY, 5, 0.5)

        # Priority 2: Use Potion
        elif shift_held and not action_taken:
            if self.potions_held > 0 and self.hero_health < self.hero_max_health:
                action_taken = True
                # Sound: Potion use
                was_low_health = self.hero_health < 2
                self.hero_health = min(self.hero_max_health, self.hero_health + 1)
                self.potions_held -= 1
                if was_low_health:
                    reward += 2.0
                self._add_particles(self.hero_pos * self.TILE_SIZE + self.TILE_SIZE / 2, 30, self.COLOR_POTION, 8, 0.8)

        # Priority 3: Movement
        elif movement != 0 and not action_taken:
            action_taken = True
            move_map = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
            move_dir = np.array(move_map[movement])
            self.last_move_dir = move_dir
            
            next_pos = self.hero_pos + move_dir
            if (0 <= next_pos[0] < self.GRID_WIDTH and 0 <= next_pos[1] < self.GRID_HEIGHT and 
                self.grid[next_pos[0], next_pos[1]] == 0):
                self.hero_pos = next_pos

        # --- Post-Action State Update & Rewards ---
        # Gold pickup
        gold_collected = [g for g in self.gold if np.array_equal(self.hero_pos, g)]
        if gold_collected:
            # Sound: Gold pickup
            self.gold = [g for g in self.gold if not np.array_equal(self.hero_pos, g)]
            reward += 5.0 * len(gold_collected)
            self.score += 10 * len(gold_collected)
            self._add_particles(self.hero_pos * self.TILE_SIZE + self.TILE_SIZE / 2, 15, self.COLOR_GOLD, 4, 0.4)

        # Potion pickup
        potion_collected = [p for p in self.potions_on_map if np.array_equal(self.hero_pos, p)]
        if potion_collected:
            # Sound: Potion pickup
            self.potions_on_map = [p for p in self.potions_on_map if not np.array_equal(self.hero_pos, p)]
            self.potions_held += len(potion_collected)
            self._add_particles(self.hero_pos * self.TILE_SIZE + self.TILE_SIZE / 2, 15, self.COLOR_POTION, 6, 0.4)

        # Movement reward
        new_dist_to_exit = np.linalg.norm(self.hero_pos - self.exit_pos)
        if new_dist_to_exit < old_dist_to_exit:
            reward += 0.1
        elif new_dist_to_exit > old_dist_to_exit:
            reward -= 0.1

        # Check for win condition
        if np.array_equal(self.hero_pos, self.exit_pos):
            # Sound: Level complete
            reward += 100.0
            self.score += 1000
            terminated = True
            self.game_over = True
            self.game_won = True

        # --- Enemy Turn ---
        if not terminated:
            for enemy in self.enemies:
                enemy_pos = enemy['pos']
                # Check for adjacency before moving
                if np.linalg.norm(enemy_pos - self.hero_pos) < 1.5:
                     # Sound: Player hurt
                    self.hero_health -= 1
                    reward -= 0.5 # Small penalty for getting hit
                    self._add_particles(self.hero_pos * self.TILE_SIZE + self.TILE_SIZE / 2, 20, self.COLOR_HERO, 5, 0.6)
                else:
                    # Move towards hero
                    diff = self.hero_pos - enemy_pos
                    move = np.zeros(2, dtype=int)
                    if abs(diff[0]) > abs(diff[1]):
                        move[0] = np.sign(diff[0])
                    elif abs(diff[1]) != 0:
                        move[1] = np.sign(diff[1])

                    next_enemy_pos = enemy_pos + move
                    is_wall = self.grid[next_enemy_pos[0], next_enemy_pos[1]] == 1
                    is_occupied = any(np.array_equal(next_enemy_pos, other['pos']) for other in self.enemies)
                    
                    if not is_wall and not is_occupied:
                        enemy['pos'] = next_enemy_pos

        # --- Final State Checks ---
        self.steps += 1
        
        if self.hero_health <= 0:
            # Sound: Player death
            reward -= 100.0
            terminated = True
            self.game_over = True
        
        if self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
        
        self._update_particles()

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Camera follows hero
        target_offset_x = self.SCREEN_WIDTH / 2 - (self.hero_pos[0] + 0.5) * self.TILE_SIZE
        target_offset_y = self.SCREEN_HEIGHT / 2 - (self.hero_pos[1] + 0.5) * self.TILE_SIZE
        # Smooth camera movement
        self.world_offset[0] += (target_offset_x - self.world_offset[0]) * 0.1
        self.world_offset[1] += (target_offset_y - self.world_offset[1]) * 0.1

        # Draw grid
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                rect = pygame.Rect(self.world_offset[0] + x * self.TILE_SIZE, self.world_offset[1] + y * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
                color = self.COLOR_WALL if self.grid[x, y] == 1 else self.COLOR_FLOOR
                pygame.draw.rect(self.screen, color, rect)

        # Draw exit
        exit_rect = pygame.Rect(self.world_offset[0] + self.exit_pos[0] * self.TILE_SIZE, self.world_offset[1] + self.exit_pos[1] * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_EXIT, exit_rect)
        pygame.gfxdraw.filled_circle(self.screen, int(exit_rect.centerx), int(exit_rect.centery), int(self.TILE_SIZE * 0.3), self.COLOR_BG)
        pygame.gfxdraw.aacircle(self.screen, int(exit_rect.centerx), int(exit_rect.centery), int(self.TILE_SIZE * 0.3), self.COLOR_BG)

        # Draw items
        for pos in self.gold:
            center_x = int(self.world_offset[0] + (pos[0] + 0.5) * self.TILE_SIZE)
            center_y = int(self.world_offset[1] + (pos[1] + 0.5) * self.TILE_SIZE)
            pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, int(self.TILE_SIZE * 0.2), self.COLOR_GOLD)
            pygame.gfxdraw.aacircle(self.screen, center_x, center_y, int(self.TILE_SIZE * 0.2), self.COLOR_GOLD)
        
        for pos in self.potions_on_map:
            center_x = int(self.world_offset[0] + (pos[0] + 0.5) * self.TILE_SIZE)
            center_y = int(self.world_offset[1] + (pos[1] + 0.5) * self.TILE_SIZE)
            r = int(self.TILE_SIZE * 0.25)
            pygame.draw.line(self.screen, self.COLOR_POTION, (center_x - r, center_y), (center_x + r, center_y), 3)
            pygame.draw.line(self.screen, self.COLOR_POTION, (center_x, center_y - r), (center_x, center_y + r), 3)

        # Draw enemies
        for enemy in self.enemies:
            center_x = int(self.world_offset[0] + (enemy['pos'][0] + 0.5) * self.TILE_SIZE)
            center_y = int(self.world_offset[1] + (enemy['pos'][1] + 0.5) * self.TILE_SIZE)
            pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, int(self.TILE_SIZE * 0.35), self.COLOR_ENEMY)
            pygame.gfxdraw.aacircle(self.screen, center_x, center_y, int(self.TILE_SIZE * 0.35), self.COLOR_ENEMY)

        # Draw hero
        hero_center_x = int(self.world_offset[0] + (self.hero_pos[0] + 0.5) * self.TILE_SIZE)
        hero_center_y = int(self.world_offset[1] + (self.hero_pos[1] + 0.5) * self.TILE_SIZE)
        pygame.gfxdraw.filled_circle(self.screen, hero_center_x, hero_center_y, int(self.TILE_SIZE * 0.4), self.COLOR_HERO)
        pygame.gfxdraw.aacircle(self.screen, hero_center_x, hero_center_y, int(self.TILE_SIZE * 0.4), self.COLOR_HERO)
        
        # Facing indicator
        facing_x = hero_center_x + self.last_move_dir[0] * self.TILE_SIZE * 0.25
        facing_y = hero_center_y + self.last_move_dir[1] * self.TILE_SIZE * 0.25
        pygame.gfxdraw.filled_circle(self.screen, int(facing_x), int(facing_y), int(self.TILE_SIZE * 0.1), self.COLOR_BG)

        # Draw particles
        for p in self.particles:
            size = int(p['size'] * (p['life'] / p['max_life']))
            if size > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(self.world_offset[0] + p['pos'][0]), int(self.world_offset[1] + p['pos'][1]), size, p['color'])

    def _render_ui(self):
        # Health bar
        bar_width = 150
        bar_height = 20
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BG, (10, 10, bar_width, bar_height))
        health_ratio = self.hero_health / self.hero_max_health
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_FG, (10, 10, int(bar_width * health_ratio), bar_height))

        # Potion counter
        potion_text = self.font_ui.render(f"x{self.potions_held}", True, self.COLOR_TEXT)
        r = int(bar_height * 0.4)
        cx, cy = 10 + bar_width + 20, 10 + bar_height // 2
        pygame.draw.line(self.screen, self.COLOR_POTION, (cx - r, cy), (cx + r, cy), 3)
        pygame.draw.line(self.screen, self.COLOR_POTION, (cx, cy - r), (cx, cy + r), 3)
        self.screen.blit(potion_text, (cx + r + 5, 10))

        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH - score_text.get_width() - 10, 10))
        
        # Game Over / Win Message
        if self.game_over:
            msg_text = "YOU DIED"
            msg_color = self.COLOR_ENEMY
            if self.game_won:
                msg_text = "VICTORY!"
                msg_color = self.COLOR_GOLD
            
            rendered_msg = self.font_msg.render(msg_text, True, msg_color)
            msg_rect = rendered_msg.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            
            # Draw a semi-transparent background for the message
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 128))
            self.screen.blit(overlay, (0, 0))
            
            self.screen.blit(rendered_msg, msg_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "health": self.hero_health,
            "potions": self.potions_held,
            "enemies_left": len(self.enemies),
        }

    def _add_particles(self, pos, count, color, size_base, life_base):
        for _ in range(count):
            angle = self.rng.uniform(0, 2 * math.pi)
            speed = self.rng.uniform(1, 5)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            life = life_base + self.rng.uniform(-0.2, 0.2)
            self.particles.append({
                'pos': list(pos),
                'vel': vel,
                'color': color,
                'size': size_base + self.rng.uniform(-2, 2),
                'life': life,
                'max_life': life
            })

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1/30.0 # Assuming 30 FPS for life decay
        self.particles = [p for p in self.particles if p['life'] > 0]

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
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

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually
    env = GameEnv()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Dungeon Crawler")
    clock = pygame.time.Clock()
    
    running = True
    while running:
        movement, space, shift = 0, 0, 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        elif keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        if keys[pygame.K_SPACE]:
            space = 1
        
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift = 1

        action = [movement, space, shift]
        
        # In a manual play loop, we only step if an action is taken.
        # This simulates the turn-based nature for a human player.
        if any(action):
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Action: {action}, Reward: {reward:.2f}, Info: {info}")
            if terminated:
                print("Game Over! Press 'R' to reset.")
        
        # We still need to render every frame for a smooth experience
        # The observation from the step is what we draw
        draw_surface = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(draw_surface, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit to 30 FPS

    env.close()