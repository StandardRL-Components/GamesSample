
# Generated: 2025-08-27T23:53:28.395022
# Source Brief: brief_03608.md
# Brief Index: 3608

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move your yellow square. Collect red gems and avoid the pink enemies."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Collect gems in a procedurally generated maze while dodging enemies. Each gem collected makes the enemies faster. Get 10 gems to win, but 5 hits from an enemy and you lose."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    MAZE_WIDTH, MAZE_HEIGHT = 21, 15  # Odd numbers for maze generation
    TILE_SIZE = 20
    GAME_AREA_WIDTH = MAZE_WIDTH * TILE_SIZE
    GAME_AREA_HEIGHT = MAZE_HEIGHT * TILE_SIZE
    OFFSET_X = (SCREEN_WIDTH - GAME_AREA_WIDTH) // 2
    OFFSET_Y = (SCREEN_HEIGHT - GAME_AREA_HEIGHT) // 2

    # Colors
    COLOR_BG = (20, 25, 40)
    COLOR_WALL = (40, 50, 80)
    COLOR_FLOOR = (30, 35, 60)
    COLOR_PLAYER = (255, 255, 0)
    COLOR_PLAYER_GLOW = (255, 255, 150)
    COLOR_GEM = (255, 20, 50)
    COLOR_GEM_SPARKLE = (255, 150, 150)
    COLOR_ENEMY = (255, 105, 180)
    COLOR_ENEMY_GLOW = (255, 180, 210)
    COLOR_GEM_SPAWN = (50, 60, 90)
    COLOR_UI_TEXT = (220, 220, 220)
    COLOR_UI_ICON = (180, 180, 180)

    # Game settings
    MAX_STEPS = 1000
    WIN_GEM_COUNT = 10
    LOSE_CONTACT_COUNT = 5
    INITIAL_ENEMY_SPEED = 5
    INITIAL_ENEMIES = 2
    INITIAL_GEMS = 3

    ENEMY_PATH = [(0, -1), (0, -1), (1, 0), (1, 0), (0, 1), (0, 1), (-1, 0), (-1, 0)] # 8-step patrol

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
        self.font_small = pygame.font.SysFont("Arial", 18, bold=True)
        self.font_large = pygame.font.SysFont("Arial", 24, bold=True)
        
        self.np_random = None

        # These will be initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.maze = []
        self.floor_tiles = []
        self.player_pos = (0, 0)
        self.gems = []
        self.enemies = []
        self.gems_collected = 0
        self.enemy_contacts = 0
        self.enemy_speed = self.INITIAL_ENEMY_SPEED
        self.particles = []

        self.reset()
        self.validate_implementation()


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
            random.seed(seed)
        else:
            if self.np_random is None:
                self.np_random = np.random.default_rng()


        self.steps = 0
        self.score = 0
        self.game_over = False
        self.gems_collected = 0
        self.enemy_contacts = 0
        self.enemy_speed = self.INITIAL_ENEMY_SPEED
        self.particles = []

        self._generate_maze()
        
        self.floor_tiles = []
        for y in range(self.MAZE_HEIGHT):
            for x in range(self.MAZE_WIDTH):
                if self.maze[y][x] == 0:
                    self.floor_tiles.append((x, y))

        # Place player
        self.player_pos = (1, 1)
        
        if self.player_pos not in self.floor_tiles:
             self.player_pos = self.floor_tiles[0]

        available_spawns = list(self.floor_tiles)
        available_spawns.remove(self.player_pos)
        
        # Place gems
        self.gems = []
        for _ in range(self.INITIAL_GEMS):
            if not available_spawns: break
            pos = random.choice(available_spawns)
            self.gems.append(pos)
            available_spawns.remove(pos)

        # Place enemies
        self.enemies = []
        far_spawns = [p for p in available_spawns if abs(p[0] - self.player_pos[0]) + abs(p[1] - self.player_pos[1]) > 5]
        if not far_spawns: far_spawns = available_spawns
        
        for _ in range(self.INITIAL_ENEMIES):
            if not far_spawns: break
            pos = random.choice(far_spawns)
            self.enemies.append({
                "pos": pos,
                "path_index": random.randint(0, len(self.ENEMY_PATH) - 1),
                "move_counter": random.randint(0, self.enemy_speed -1)
            })
            far_spawns.remove(pos)
            if pos in available_spawns: available_spawns.remove(pos)


        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        reward = 0
        
        # --- Pre-move state for reward calculation ---
        dist_to_gem_before = self._get_min_dist(self.player_pos, self.gems)
        dist_to_enemy_before = self._get_min_dist(self.player_pos, [e['pos'] for e in self.enemies])

        # --- Player Movement ---
        if movement != 0:
            dx, dy = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)][movement]
            new_x, new_y = self.player_pos[0] + dx, self.player_pos[1] + dy
            if 0 <= new_x < self.MAZE_WIDTH and 0 <= new_y < self.MAZE_HEIGHT and self.maze[new_y][new_x] == 0:
                self.player_pos = (new_x, new_y)
        
        # --- Enemy Movement ---
        for enemy in self.enemies:
            enemy['move_counter'] += 1
            if enemy['move_counter'] >= self.enemy_speed:
                enemy['move_counter'] = 0
                dx, dy = self.ENEMY_PATH[enemy['path_index']]
                new_ex, new_ey = enemy['pos'][0] + dx, enemy['pos'][1] + dy
                
                if 0 <= new_ex < self.MAZE_WIDTH and 0 <= new_ey < self.MAZE_HEIGHT and self.maze[new_ey][new_ex] == 0:
                    enemy['pos'] = (new_ex, new_ey)
                
                enemy['path_index'] = (enemy['path_index'] + 1) % len(self.ENEMY_PATH)

        # --- Interactions and State Updates ---
        if self.player_pos in self.gems:
            reward += 10
            self.gems.remove(self.player_pos)
            self.gems_collected += 1
            self._spawn_particles(self.player_pos, self.COLOR_GEM_SPARKLE)
            
            # Respawn gem
            spawn_options = [s for s in self.floor_tiles if s != self.player_pos and s not in self.gems and s not in [e['pos'] for e in self.enemies]]
            if spawn_options:
                self.gems.append(random.choice(spawn_options))

            # Increase enemy speed
            self.enemy_speed = max(1, self.INITIAL_ENEMY_SPEED - (self.gems_collected // 2))

        for enemy in self.enemies:
            if self.player_pos == enemy['pos']:
                self.enemy_contacts += 1
                reward -= 5
                self._spawn_particles(self.player_pos, self.COLOR_ENEMY_GLOW, 5)
                break # Only one contact per step

        # --- Continuous Reward Calculation ---
        dist_to_gem_after = self._get_min_dist(self.player_pos, self.gems)
        dist_to_enemy_after = self._get_min_dist(self.player_pos, [e['pos'] for e in self.enemies])
        
        if dist_to_gem_after < dist_to_gem_before:
            reward += 1.0
        elif dist_to_gem_after > dist_to_gem_before:
            reward -= 0.1

        if dist_to_enemy_after < dist_to_enemy_before:
            reward -= 0.1
        
        # --- Update state and check for termination ---
        self.steps += 1
        terminated = self._check_termination()

        if terminated:
            if self.gems_collected >= self.WIN_GEM_COUNT:
                reward += 100
            if self.enemy_contacts >= self.LOSE_CONTACT_COUNT:
                reward -= 100
            self.game_over = True
        
        self.score += reward
        self._update_particles()

        return self._get_observation(), reward, terminated, False, self._get_info()
    
    def _get_min_dist(self, pos, targets):
        if not targets:
            return float('inf')
        return min(abs(pos[0] - t[0]) + abs(pos[1] - t[1]) for t in targets)

    def _check_termination(self):
        return (
            self.gems_collected >= self.WIN_GEM_COUNT or
            self.enemy_contacts >= self.LOSE_CONTACT_COUNT or
            self.steps >= self.MAX_STEPS
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
            "gems_collected": self.gems_collected,
            "enemy_contacts": self.enemy_contacts,
        }

    def _render_game(self):
        # Draw floor and walls
        for y in range(self.MAZE_HEIGHT):
            for x in range(self.MAZE_WIDTH):
                rect = pygame.Rect(self.OFFSET_X + x * self.TILE_SIZE, self.OFFSET_Y + y * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
                if self.maze[y][x] == 1:
                    pygame.draw.rect(self.screen, self.COLOR_WALL, rect)
                else:
                    pygame.draw.rect(self.screen, self.COLOR_FLOOR, rect)

        # Draw gem spawn points
        for x, y in self.floor_tiles:
            cx = int(self.OFFSET_X + (x + 0.5) * self.TILE_SIZE)
            cy = int(self.OFFSET_Y + (y + 0.5) * self.TILE_SIZE)
            pygame.gfxdraw.filled_circle(self.screen, cx, cy, 3, self.COLOR_GEM_SPAWN)
        
        # Draw gems
        gem_size_mod = (math.sin(self.steps * 0.2) + 1) / 2  # 0 to 1
        gem_base_size = self.TILE_SIZE * 0.4
        for x, y in self.gems:
            cx = self.OFFSET_X + (x + 0.5) * self.TILE_SIZE
            cy = self.OFFSET_Y + (y + 0.5) * self.TILE_SIZE
            size = gem_base_size + gem_size_mod * 3
            points = [
                (cx, cy - size),
                (cx + size, cy),
                (cx, cy + size),
                (cx - size, cy)
            ]
            pygame.draw.polygon(self.screen, self.COLOR_GEM, [(int(px), int(py)) for px, py in points])
            pygame.draw.aalines(self.screen, self.COLOR_GEM_SPARKLE, True, [(int(px), int(py)) for px, py in points])

        # Draw enemies
        enemy_size_mod = (math.sin(self.steps * 0.1) + 1) / 2
        enemy_radius = self.TILE_SIZE * 0.35 + enemy_size_mod * 2
        for enemy in self.enemies:
            ex, ey = enemy['pos']
            cx = int(self.OFFSET_X + (ex + 0.5) * self.TILE_SIZE)
            cy = int(self.OFFSET_Y + (ey + 0.5) * self.TILE_SIZE)
            pygame.gfxdraw.filled_circle(self.screen, cx, cy, int(enemy_radius), self.COLOR_ENEMY)
            pygame.gfxdraw.aacircle(self.screen, cx, cy, int(enemy_radius), self.COLOR_ENEMY_GLOW)

        # Draw player
        px, py = self.player_pos
        player_rect = pygame.Rect(self.OFFSET_X + px * self.TILE_SIZE, self.OFFSET_Y + py * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
        
        # Player glow effect
        glow_size = self.TILE_SIZE * 1.5 + (math.sin(self.steps * 0.15) * 5)
        glow_surf = pygame.Surface((glow_size, glow_size), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, (*self.COLOR_PLAYER_GLOW, 50), (glow_size/2, glow_size/2), glow_size/2)
        self.screen.blit(glow_surf, (player_rect.centerx - glow_size/2, player_rect.centery - glow_size/2))
        
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect.inflate(-4, -4))

        # Draw particles
        for p in self.particles:
            p_color = (*p['color'], int(255 * (p['life'] / p['max_life'])))
            p_surf = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
            pygame.draw.circle(p_surf, p_color, (p['size'], p['size']), p['size'])
            self.screen.blit(p_surf, (int(p['pos'][0] - p['size']), int(p['pos'][1] - p['size'])))

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (20, 10))

        # Gems collected
        gem_count_text = self.font_large.render(f"{self.gems_collected} / {self.WIN_GEM_COUNT}", True, self.COLOR_UI_TEXT)
        text_rect = gem_count_text.get_rect(topright=(self.SCREEN_WIDTH - 20, 10))
        self.screen.blit(gem_count_text, text_rect)
        # Gem Icon
        icon_points = [ (text_rect.left - 20, 22), (text_rect.left - 12, 15), (text_rect.left - 20, 8), (text_rect.left - 28, 15) ]
        pygame.draw.polygon(self.screen, self.COLOR_GEM, icon_points)
        
        # Enemy contacts
        contact_count_text = self.font_large.render(f"{self.enemy_contacts} / {self.LOSE_CONTACT_COUNT}", True, self.COLOR_UI_TEXT)
        text_rect = contact_count_text.get_rect(bottomright=(self.SCREEN_WIDTH - 20, self.SCREEN_HEIGHT - 10))
        self.screen.blit(contact_count_text, text_rect)
        # Enemy Icon
        pygame.draw.circle(self.screen, self.COLOR_ENEMY, (text_rect.left - 20, text_rect.centery), 8)

        # Game Over Message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            message = "YOU WIN!" if self.gems_collected >= self.WIN_GEM_COUNT else "GAME OVER"
            msg_text = self.font_large.render(message, True, self.COLOR_PLAYER if message == "YOU WIN!" else self.COLOR_GEM)
            msg_rect = msg_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(msg_text, msg_rect)

    def _generate_maze(self):
        # Using randomized Prim's algorithm
        self.maze = np.ones((self.MAZE_HEIGHT, self.MAZE_WIDTH), dtype=np.uint8)
        start_x, start_y = (1, 1)
        self.maze[start_y, start_x] = 0
        
        frontiers = []
        for dx, dy in [(0, 2), (0, -2), (2, 0), (-2, 0)]:
            nx, ny = start_x + dx, start_y + dy
            if 0 <= nx < self.MAZE_WIDTH and 0 <= ny < self.MAZE_HEIGHT:
                frontiers.append((nx, ny))

        while frontiers:
            fx, fy = random.choice(frontiers)
            frontiers.remove((fx, fy))

            neighbors = []
            for dx, dy in [(0, 2), (0, -2), (2, 0), (-2, 0)]:
                nx, ny = fx + dx, fy + dy
                if 0 <= nx < self.MAZE_WIDTH and 0 <= ny < self.MAZE_HEIGHT and self.maze[ny, nx] == 0:
                    neighbors.append((nx, ny))
            
            if neighbors:
                nx, ny = random.choice(neighbors)
                self.maze[fy, fx] = 0
                self.maze[(fy + ny) // 2, (fx + nx) // 2] = 0

                for dx, dy in [(0, 2), (0, -2), (2, 0), (-2, 0)]:
                    nfx, nfy = fx + dx, fy + dy
                    if 0 <= nfx < self.MAZE_WIDTH and 0 <= nfy < self.MAZE_HEIGHT and self.maze[nfy, nfx] == 1:
                        if (nfx, nfy) not in frontiers:
                            frontiers.append((nfx, nfy))
        
        # Add some loops to make it less claustrophobic
        for _ in range((self.MAZE_WIDTH * self.MAZE_HEIGHT) // 20):
            x, y = random.randint(1, self.MAZE_WIDTH-2), random.randint(1, self.MAZE_HEIGHT-2)
            if self.maze[y, x] == 1:
                self.maze[y, x] = 0

    def _spawn_particles(self, pos_tile, color, count=10):
        cx = self.OFFSET_X + (pos_tile[0] + 0.5) * self.TILE_SIZE
        cy = self.OFFSET_Y + (pos_tile[1] + 0.5) * self.TILE_SIZE
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            life = random.randint(10, 20)
            self.particles.append({
                'pos': [cx, cy],
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': life,
                'max_life': life,
                'color': color,
                'size': random.uniform(2, 5)
            })

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            p['vel'][0] *= 0.95 # friction
            p['vel'][1] *= 0.95

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
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
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    import os
    
    # To run with display, set a video driver.
    # If you are running on a headless server, you can comment this out.
    if os.name == 'posix':
        os.environ["SDL_VIDEODRIVER"] = "x11"
    
    env = GameEnv()
    
    try:
        pygame.quit() # Quit the headless instance
        pygame.init() # Re-init with video
        screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
        pygame.display.set_caption("Gem Maze")
        clock = pygame.time.Clock()

        obs, info = env.reset()
        done = False
        
        print(GameEnv.game_description)
        print(GameEnv.user_guide)
        
        last_action_time = pygame.time.get_ticks()
        action_delay = 100 # ms between actions

        while not done:
            current_time = pygame.time.get_ticks()
            movement_action = 0 
            
            # --- Event Handling ---
            events = pygame.event.get()
            for event in events:
                if event.type == pygame.QUIT:
                    done = True
                if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    print("--- RESETTING GAME ---")
                    obs, info = env.reset()

            # --- Player Control ---
            if current_time - last_action_time > action_delay:
                keys = pygame.key.get_pressed()
                if keys[pygame.K_UP] or keys[pygame.K_w]:
                    movement_action = 1
                elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
                    movement_action = 2
                elif keys[pygame.K_LEFT] or keys[pygame.K_a]:
                    movement_action = 3
                elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
                    movement_action = 4
                
                # --- Step the Environment ---
                if movement_action != 0:
                    last_action_time = current_time
                    action = [movement_action, 0, 0] # space/shift not used
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    
                    print(f"Step: {info['steps']}, Score: {int(info['score'])}, Reward: {reward:.1f}, Gems: {info['gems_collected']}, Hits: {info['enemy_contacts']}")

            # --- Render the Environment to the Screen ---
            frame = np.transpose(obs, (1, 0, 2))
            surf = pygame.surfarray.make_surface(frame)
            screen.blit(surf, (0, 0))
            
            pygame.display.flip()
            clock.tick(60) 

    finally:
        env.close()