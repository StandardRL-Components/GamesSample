
# Generated: 2025-08-28T05:04:10.462188
# Source Brief: brief_02509.md
# Brief Index: 2509

        
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
        "Controls: Arrow keys to move your character (blue square). "
        "Collect the shimmering crystals to score points."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Collect shimmering crystals in an isometric world while dodging patrolling enemies to amass a glittering hoard."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_SIZE = 20
    TILE_WIDTH = 28
    TILE_HEIGHT = 14
    MAX_STEPS = 1000
    NUM_CRYSTALS = 25
    NUM_ENEMIES = 3

    # Colors
    COLOR_BG = (25, 45, 40)
    COLOR_GRID = (45, 75, 70)
    COLOR_PLAYER = (100, 200, 255)
    COLOR_PLAYER_OUTLINE = (220, 255, 255)
    COLOR_ENEMY = (255, 80, 80)
    COLOR_ENEMY_OUTLINE = (255, 150, 150)
    CRYSTAL_COLORS = [
        (100, 150, 255),  # Blue
        (100, 255, 150),  # Green
        (255, 100, 200),  # Purple
        (255, 255, 100),  # Yellow
    ]
    COLOR_UI_TEXT = (230, 230, 230)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 48, bold=True)

        # Game state variables
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_pos = [0, 0]
        self.crystals = []
        self.enemies = []
        self.particles = []
        self.crystals_collected = 0
        self.last_dist_to_crystal = 0
        self.last_dist_to_enemy = 0

        # Calculate isometric projection origin
        self.origin_x = self.SCREEN_WIDTH // 2
        self.origin_y = self.SCREEN_HEIGHT // 2 - (self.GRID_SIZE * self.TILE_HEIGHT) // 2 + 50

        # Initialize state variables
        self.reset()
        
        # Run validation check
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.crystals_collected = 0
        self.particles = []

        # --- Place Game Objects ---
        occupied_cells = set()

        # Player
        self.player_pos = [self.GRID_SIZE // 2, self.GRID_SIZE // 2]
        occupied_cells.add(tuple(self.player_pos))

        # Enemies
        self.enemies = []
        enemy_paths = [
            self._generate_path(3, 3, 10, 'rect'),
            self._generate_path(self.GRID_SIZE - 4, 3, 12, 'rect'),
            self._generate_path(self.GRID_SIZE // 2, self.GRID_SIZE - 2, 8, 'circle')
        ]
        for path in enemy_paths:
            start_pos = path[0]
            self.enemies.append({'path': path, 'path_index': 0, 'pos': list(start_pos)})
            occupied_cells.add(start_pos)

        # Crystals
        self.crystals = []
        for _ in range(self.NUM_CRYSTALS):
            while True:
                pos = (self.np_random.integers(0, self.GRID_SIZE), self.np_random.integers(0, self.GRID_SIZE))
                if pos not in occupied_cells:
                    occupied_cells.add(pos)
                    color_idx = self.np_random.integers(0, len(self.CRYSTAL_COLORS))
                    anim_offset = self.np_random.random() * math.pi * 2
                    self.crystals.append({'pos': list(pos), 'color_idx': color_idx, 'anim_offset': anim_offset})
                    break
        
        # Pre-calculate initial distances for reward
        self.last_dist_to_crystal, _ = self._find_nearest_item(self.player_pos, [c['pos'] for c in self.crystals])
        self.last_dist_to_enemy, _ = self._find_nearest_item(self.player_pos, [e['pos'] for e in self.enemies])

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0
        movement = action[0]

        # --- Update Player Position ---
        prev_player_pos = list(self.player_pos)
        if movement == 1:  # Up (NW)
            self.player_pos[0] -= 1
        elif movement == 2:  # Down (SE)
            self.player_pos[0] += 1
        elif movement == 3:  # Left (SW)
            self.player_pos[1] -= 1
        elif movement == 4:  # Right (NE)
            self.player_pos[1] += 1

        # Clamp player to grid
        self.player_pos[0] = np.clip(self.player_pos[0], 0, self.GRID_SIZE - 1)
        self.player_pos[1] = np.clip(self.player_pos[1], 0, self.GRID_SIZE - 1)
        
        # --- Update Enemies ---
        for enemy in self.enemies:
            enemy['path_index'] = (enemy['path_index'] + 1) % len(enemy['path'])
            enemy['pos'] = list(enemy['path'][enemy['path_index']])

        # --- Reward for Movement ---
        dist_to_crystal, _ = self._find_nearest_item(self.player_pos, [c['pos'] for c in self.crystals])
        dist_to_enemy, _ = self._find_nearest_item(self.player_pos, [e['pos'] for e in self.enemies])

        if dist_to_crystal is not None:
            reward += 0.1 * (self.last_dist_to_crystal - dist_to_crystal)
            self.last_dist_to_crystal = dist_to_crystal
        
        if dist_to_enemy is not None:
            reward -= 0.2 * (self.last_dist_to_enemy - dist_to_enemy) # Negative reward for getting closer
            self.last_dist_to_enemy = dist_to_enemy

        if movement == 0:
            reward -= 0.05
        
        # --- Check Crystal Collection ---
        collected_crystal_idx = -1
        for i, crystal in enumerate(self.crystals):
            if crystal['pos'] == self.player_pos:
                collected_crystal_idx = i
                break
        
        if collected_crystal_idx != -1:
            # Sound: crystal_get.wav
            self.crystals.pop(collected_crystal_idx)
            self.crystals_collected += 1
            
            base_reward = 1.0
            self.score += int(base_reward * 10)
            
            # Bonus for risky collection
            if dist_to_enemy is not None and dist_to_enemy <= 3:
                base_reward += 2.0
                self.score += 20

            reward += base_reward
            self._spawn_particles(self.player_pos, self.CRYSTAL_COLORS[crystal['color_idx']])

        # --- Update Particles ---
        self._update_particles()
        
        # --- Check Termination Conditions ---
        terminated = False
        # 1. Collision with enemy
        for enemy in self.enemies:
            if enemy['pos'] == self.player_pos:
                # Sound: player_hit.wav
                self.game_over = True
                terminated = True
                reward = -100.0
                self.score -= 100
                break
        
        if not terminated:
            # 2. All crystals collected (Win)
            if not self.crystals:
                # Sound: game_win.wav
                self.game_over = True
                terminated = True
                reward = 100.0
                self.score += 1000
            # 3. Max steps reached
            elif self.steps >= self.MAX_STEPS -1:
                self.game_over = True
                terminated = True
                self.score -= 50

        self.steps += 1

        return self._get_observation(), reward, terminated, False, self._get_info()

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
            "crystals_collected": self.crystals_collected,
            "crystals_remaining": len(self.crystals),
        }

    def close(self):
        pygame.quit()

    # --- Helper & Rendering Methods ---

    def _iso_to_screen(self, x, y):
        screen_x = self.origin_x + (y - x) * self.TILE_WIDTH / 2
        screen_y = self.origin_y + (y + x) * self.TILE_HEIGHT / 2
        return int(screen_x), int(screen_y)

    def _render_game(self):
        # --- Render Grid ---
        for i in range(self.GRID_SIZE + 1):
            start_p1 = self._iso_to_screen(i, 0)
            end_p1 = self._iso_to_screen(i, self.GRID_SIZE)
            pygame.draw.aaline(self.screen, self.COLOR_GRID, start_p1, end_p1)
            
            start_p2 = self._iso_to_screen(0, i)
            end_p2 = self._iso_to_screen(self.GRID_SIZE, i)
            pygame.draw.aaline(self.screen, self.COLOR_GRID, start_p2, end_p2)

        # --- Render Objects (sorted by screen Y for correct layering) ---
        render_queue = []
        # Add crystals
        for c in self.crystals:
            screen_pos = self._iso_to_screen(c['pos'][0], c['pos'][1])
            render_queue.append((screen_pos[1], 'crystal', c))
        # Add enemies
        for e in self.enemies:
            screen_pos = self._iso_to_screen(e['pos'][0], e['pos'][1])
            render_queue.append((screen_pos[1], 'enemy', e))
        # Add player
        screen_pos = self._iso_to_screen(self.player_pos[0], self.player_pos[1])
        render_queue.append((screen_pos[1], 'player', {}))
        
        render_queue.sort(key=lambda item: item[0])
        
        for _, type, data in render_queue:
            if type == 'crystal':
                self._render_crystal(data)
            elif type == 'enemy':
                self._render_enemy(data)
            elif type == 'player':
                self._render_player()

        # --- Render Particles (on top of everything) ---
        self._render_particles()

    def _render_crystal(self, crystal):
        pos = crystal['pos']
        color = self.CRYSTAL_COLORS[crystal['color_idx']]
        anim_offset = crystal['anim_offset']
        screen_pos = self._iso_to_screen(pos[0], pos[1])
        
        # Sparkle effect
        t = (self.steps * 0.1 + anim_offset)
        sparkle_size = 4 + 2 * math.sin(t)
        sparkle_alpha = 150 + 100 * math.sin(t * 0.7)

        points = [
            (screen_pos[0], screen_pos[1] - self.TILE_HEIGHT * 0.5),
            (screen_pos[0] + self.TILE_WIDTH * 0.3, screen_pos[1]),
            (screen_pos[0], screen_pos[1] + self.TILE_HEIGHT * 0.5),
            (screen_pos[0] - self.TILE_WIDTH * 0.3, screen_pos[1]),
        ]
        pygame.gfxdraw.aapolygon(self.screen, points, color)
        pygame.gfxdraw.filled_polygon(self.screen, points, color)
        
        # Glow
        pygame.gfxdraw.filled_circle(self.screen, screen_pos[0], screen_pos[1] - 5, int(sparkle_size), (*color, int(sparkle_alpha)))


    def _render_enemy(self, enemy):
        pos = enemy['pos']
        screen_pos = self._iso_to_screen(pos[0], pos[1])
        
        # Pulsing effect
        pulse = 1.0 + 0.1 * math.sin(self.steps * 0.2)
        w = int(self.TILE_WIDTH * 0.6 * pulse)
        h = int(self.TILE_HEIGHT * 0.6 * pulse)
        
        points = [
            (screen_pos[0], screen_pos[1] - h),
            (screen_pos[0] + w, screen_pos[1]),
            (screen_pos[0], screen_pos[1] + h),
            (screen_pos[0] - w, screen_pos[1]),
        ]
        
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_ENEMY_OUTLINE)
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_ENEMY)

    def _render_player(self):
        screen_pos = self._iso_to_screen(self.player_pos[0], self.player_pos[1])
        w, h = self.TILE_WIDTH * 0.7, self.TILE_HEIGHT * 0.7
        points = [
            (screen_pos[0], screen_pos[1] - h),
            (screen_pos[0] + w, screen_pos[1]),
            (screen_pos[0], screen_pos[1] + h),
            (screen_pos[0] - w, screen_pos[1]),
        ]
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER_OUTLINE)
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)

    def _render_ui(self):
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        crystal_text = self.font_ui.render(f"CRYSTALS: {self.crystals_collected}/{self.NUM_CRYSTALS}", True, self.COLOR_UI_TEXT)
        self.screen.blit(crystal_text, (10, 35))

        if self.game_over:
            msg = "VICTORY!" if not self.crystals else "GAME OVER"
            color = (150, 255, 150) if msg == "VICTORY!" else (255, 150, 150)
            end_text = self.font_game_over.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2))
            self.screen.blit(end_text, text_rect)

    def _spawn_particles(self, grid_pos, color):
        screen_pos = self._iso_to_screen(grid_pos[0], grid_pos[1])
        for _ in range(15):
            angle = self.np_random.random() * 2 * math.pi
            speed = 1 + self.np_random.random() * 2
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            life = 20 + self.np_random.integers(0, 10)
            self.particles.append({'pos': list(screen_pos), 'vel': vel, 'life': life, 'color': color})
            
    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]
        
    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p['life'] / 30))
            if alpha > 0:
                pygame.gfxdraw.filled_circle(
                    self.screen, int(p['pos'][0]), int(p['pos'][1]), 2, (*p['color'], alpha)
                )

    def _generate_path(self, start_x, start_y, size, shape='rect'):
        path = []
        x, y = start_x, start_y
        if shape == 'rect':
            for _ in range(size): y += 1; path.append((x, y))
            for _ in range(size): x += 1; path.append((x, y))
            for _ in range(size): y -= 1; path.append((x, y))
            for _ in range(size): x -= 1; path.append((x, y))
        elif shape == 'circle':
            for i in range(size * 4):
                angle = 2 * math.pi * i / (size * 4)
                px = int(start_x + size / 2 * math.cos(angle))
                py = int(start_y + size / 2 * math.sin(angle))
                if not path or path[-1] != (px, py):
                    path.append((px, py))
        return [
            (np.clip(p[0], 0, self.GRID_SIZE-1), np.clip(p[1], 0, self.GRID_SIZE-1)) 
            for p in path
        ]

    def _calculate_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _find_nearest_item(self, source_pos, item_positions):
        if not item_positions:
            return None, -1
        
        min_dist = float('inf')
        nearest_idx = -1
        for i, item_pos in enumerate(item_positions):
            dist = self._calculate_distance(source_pos, item_pos)
            if dist < min_dist:
                min_dist = dist
                nearest_idx = i
        return min_dist, nearest_idx
        
    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation.
        """
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

if __name__ == "__main__":
    # This block allows you to play the game manually
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Create a window to display the game
    pygame.display.set_caption("Crystal Collector")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    running = True
    total_reward = 0
    
    print("\n" + "="*30)
    print("      MANUAL PLAY MODE")
    print("="*30)
    print(GameEnv.user_guide)
    
    while running:
        action = [0, 0, 0] # Default action: no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()
                total_reward = 0
                done = False

        if not done:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]:
                action[0] = 1
            elif keys[pygame.K_DOWN]:
                action[0] = 2
            elif keys[pygame.K_LEFT]:
                action[0] = 3
            elif keys[pygame.K_RIGHT]:
                action[0] = 4
            
            # The brief specifies a turn-based game, so we only step on a keypress
            if action[0] != 0:
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                total_reward += reward
                print(f"Step: {info['steps']}, Action: {action[0]}, Reward: {reward:.2f}, Total Reward: {total_reward:.2f}, Done: {done}")

        # Render the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        env.clock.tick(15) # Limit frame rate for manual play

    env.close()