import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ↑↓←→ to move. Collect all the gems to win!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Grab gems, avoid enemies, and maximize your score in this fast-paced isometric arcade game."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_WIDTH = 8
    GRID_HEIGHT = 8
    NUM_GEMS = 25
    NUM_ENEMIES = 3
    MAX_STEPS = 1000

    # Colors
    COLOR_BG = (15, 18, 32)
    COLOR_GRID = (30, 35, 60)
    COLOR_PLAYER = (64, 224, 208)
    COLOR_PLAYER_GLOW = (64, 224, 208, 50)
    COLOR_ENEMY = (190, 20, 60)
    COLOR_ENEMY_GLOW = (190, 20, 60, 50)
    COLOR_GEM_PALETTE = [(255, 215, 0), (255, 69, 0), (50, 205, 50), (138, 43, 226)] # Gold, Red, Green, Purple
    COLOR_TEXT = (240, 240, 255)
    COLOR_MULTIPLIER = (255, 215, 0)

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
        self.font_main = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 28)
        
        # Isometric projection parameters
        self.tile_width = 32
        self.tile_height = 16
        self.origin_x = self.SCREEN_WIDTH // 2
        self.origin_y = self.SCREEN_HEIGHT // 2 - self.GRID_HEIGHT * self.tile_height

        # Initialize state variables
        self.player_pos = None
        self.enemies = []
        self.gems = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.gems_collected = 0
        self.game_over = False
        self.win = False
        self.score_multiplier = 1
        self.multiplier_timer = 0
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.gems_collected = 0
        self.game_over = False
        self.win = False
        self.particles = []
        self.score_multiplier = 1
        self.multiplier_timer = 0

        self.player_pos = np.array([self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2])
        
        self._spawn_gems()
        self._spawn_enemies()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        movement = action[0]
        
        reward = 0.0
        
        # --- Update Game Logic ---
        if not self.game_over:
            # Store pre-move state for reward calculation
            old_player_pos = self.player_pos.copy()
            
            # 1. Update Player
            self._update_player(movement)
            
            # 2. Update Enemies
            self._update_enemies()
            
            # 3. Handle Collisions & Rewards
            gem_collected_this_step, gem_reward = self._handle_gem_collisions()
            reward += gem_reward
            if gem_collected_this_step:
                # sound_effect: gem_collect.wav
                if self.multiplier_timer > 0:
                    self.score_multiplier = min(self.score_multiplier + 1, 10)
                else:
                    self.score_multiplier = 1
                self.multiplier_timer = 60 # 2 seconds at 30fps
            elif self.multiplier_timer > 0:
                self.multiplier_timer -= 1
                if self.multiplier_timer == 0:
                    self.score_multiplier = 1
            
            # 4. Calculate distance-based reward
            reward += self._calculate_distance_reward(old_player_pos, self.player_pos)

            # 5. Check for enemy collision
            enemy_collision, enemy_reward = self._handle_enemy_collisions()
            if enemy_collision:
                # sound_effect: player_hit.wav
                reward = enemy_reward # Override other rewards
                self.game_over = True

        # 6. Update particles
        self._update_particles()
        
        # 7. Check termination conditions
        self.steps += 1
        if self.gems_collected == self.NUM_GEMS:
            self.win = True
            self.game_over = True
            reward = 100.0
            # sound_effect: win_jingle.wav
        
        terminated = self.game_over
        truncated = self.steps >= self.MAX_STEPS
        if truncated:
            self.game_over = True
        
        return (
            self._get_observation(),
            float(reward),
            terminated,
            truncated,
            self._get_info()
        )
    
    def _grid_to_screen(self, grid_pos):
        x, y = grid_pos
        screen_x = self.origin_x + (x - y) * self.tile_width
        screen_y = self.origin_y + (x + y) * self.tile_height
        return int(screen_x), int(screen_y)

    def _spawn_gems(self):
        self.gems = []
        possible_coords = [(x, y) for x in range(self.GRID_WIDTH) for y in range(self.GRID_HEIGHT)]
        possible_coords.remove(tuple(self.player_pos))
        
        gem_indices = self.np_random.choice(len(possible_coords), self.NUM_GEMS, replace=False)
        for i in gem_indices:
            pos = np.array(possible_coords[i])
            color = self.COLOR_GEM_PALETTE[self.np_random.integers(0, len(self.COLOR_GEM_PALETTE))]
            self.gems.append({"pos": pos, "color": color})

    def _spawn_enemies(self):
        self.enemies = []
        # Enemy 1: Horizontal patrol
        self.enemies.append({
            "pos": np.array([0, 0]),
            "path": [np.array([x, 0]) for x in range(self.GRID_WIDTH)] + [np.array([x, 0]) for x in range(self.GRID_WIDTH - 2, 0, -1)],
            "path_idx": 0, "speed": 15
        })
        # Enemy 2: Vertical patrol
        self.enemies.append({
            "pos": np.array([self.GRID_WIDTH-1, self.GRID_HEIGHT-1]),
            "path": [np.array([self.GRID_WIDTH-1, y]) for y in range(self.GRID_HEIGHT)] + [np.array([self.GRID_WIDTH-1, y]) for y in range(self.GRID_HEIGHT - 2, 0, -1)],
            "path_idx": 0, "speed": 20
        })
        # Enemy 3: Square patrol
        path3 = ([np.array([x, 2]) for x in range(2, 6)] + 
                 [np.array([5, y]) for y in range(3, 6)] + 
                 [np.array([x, 5]) for x in range(4, 1, -1)] +
                 [np.array([2, y]) for y in range(4, 2, -1)])
        self.enemies.append({
            "pos": np.array([2, 2]),
            "path": path3,
            "path_idx": 0, "speed": 10
        })

    def _update_player(self, movement):
        if movement == 1: # Up (Iso Up-Left)
            self.player_pos[1] = max(0, self.player_pos[1] - 1)
        elif movement == 2: # Down (Iso Down-Right)
            self.player_pos[1] = min(self.GRID_HEIGHT - 1, self.player_pos[1] + 1)
        elif movement == 3: # Left (Iso Down-Left)
            self.player_pos[0] = max(0, self.player_pos[0] - 1)
        elif movement == 4: # Right (Iso Up-Right)
            self.player_pos[0] = min(self.GRID_WIDTH - 1, self.player_pos[0] + 1)
        # movement == 0 is no-op

    def _update_enemies(self):
        for enemy in self.enemies:
            if self.steps % enemy["speed"] == 0:
                enemy["path_idx"] = (enemy["path_idx"] + 1) % len(enemy["path"])
                enemy["pos"] = enemy["path"][enemy["path_idx"]]

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1
            p['radius'] = max(0, p['radius'] * 0.95)
            
    def _create_particles(self, pos, color, count=20):
        screen_pos = self._grid_to_screen(pos)
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                'pos': np.array([float(screen_pos[0]), float(screen_pos[1])]),
                'vel': np.array([math.cos(angle) * speed, math.sin(angle) * speed]),
                'life': self.np_random.integers(15, 30),
                'color': color,
                'radius': self.np_random.uniform(2, 5)
            })

    def _handle_gem_collisions(self):
        collected_gem_index = -1
        for i, gem in enumerate(self.gems):
            if np.array_equal(self.player_pos, gem["pos"]):
                collected_gem_index = i
                break
        
        if collected_gem_index != -1:
            collected_gem = self.gems.pop(collected_gem_index)
            self.gems_collected += 1
            self.score += 10 * self.score_multiplier
            self._create_particles(collected_gem["pos"], collected_gem["color"])
            return True, 10.0
        return False, 0.0

    def _handle_enemy_collisions(self):
        for enemy in self.enemies:
            if np.array_equal(self.player_pos, enemy["pos"]):
                self._create_particles(self.player_pos, self.COLOR_ENEMY, 50)
                return True, -100.0
        return False, 0.0

    def _find_nearest_gem_dist(self, pos):
        if not self.gems:
            return 0
        distances = [np.linalg.norm(pos - gem["pos"]) for gem in self.gems]
        return min(distances)

    def _calculate_distance_reward(self, old_pos, new_pos):
        if not self.gems:
            return 0.0
        
        old_dist = self._find_nearest_gem_dist(old_pos)
        new_dist = self._find_nearest_gem_dist(new_pos)

        if new_dist < old_dist:
            return 1.0
        elif new_dist > old_dist:
            return -0.1
        return 0.0

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # 1. Render Grid
        for y in range(self.GRID_HEIGHT + 1):
            start = self._grid_to_screen((0, y))
            end = self._grid_to_screen((self.GRID_WIDTH, y))
            pygame.draw.aaline(self.screen, self.COLOR_GRID, start, end)
        for x in range(self.GRID_WIDTH + 1):
            start = self._grid_to_screen((x, 0))
            end = self._grid_to_screen((x, self.GRID_HEIGHT))
            pygame.draw.aaline(self.screen, self.COLOR_GRID, start, end)

        # 2. Render Gems
        pulse = (math.sin(self.steps * 0.1) + 1) / 2 * 2 # Pulse from 0 to 2
        for gem in self.gems:
            screen_pos = self._grid_to_screen(gem["pos"])
            size = int(self.tile_height / 2 + pulse)
            points = [
                (screen_pos[0], screen_pos[1] - size),
                (screen_pos[0] + size, screen_pos[1]),
                (screen_pos[0], screen_pos[1] + size),
                (screen_pos[0] - size, screen_pos[1])
            ]
            pygame.gfxdraw.aapolygon(self.screen, points, gem["color"])
            pygame.gfxdraw.filled_polygon(self.screen, points, gem["color"])

        # 3. Render Enemies
        for enemy in self.enemies:
            self._render_iso_cube(enemy["pos"], self.COLOR_ENEMY, self.COLOR_ENEMY_GLOW)

        # 4. Render Player
        if not (self.game_over and not self.win):
            self._render_iso_cube(self.player_pos, self.COLOR_PLAYER, self.COLOR_PLAYER_GLOW)
            
        # 5. Render Particles
        for p in self.particles:
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            pygame.draw.circle(self.screen, p['color'], pos, int(p['radius']))

    def _render_iso_cube(self, pos, color, glow_color):
        screen_pos = self._grid_to_screen(pos)
        
        # Glow effect
        glow_radius = int(self.tile_width * 1.2)
        glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, glow_color, (glow_radius, glow_radius), glow_radius)
        self.screen.blit(glow_surf, (screen_pos[0] - glow_radius, screen_pos[1] - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)

        # Cube
        h = self.tile_height
        w = self.tile_width
        top_points = [
            (screen_pos[0], screen_pos[1]),
            (screen_pos[0] + w, screen_pos[1] + h),
            (screen_pos[0], screen_pos[1] + 2*h),
            (screen_pos[0] - w, screen_pos[1] + h)
        ]
        side_color = tuple(c * 0.7 for c in color)
        
        # Draw sides first
        pygame.gfxdraw.filled_polygon(self.screen, [(top_points[2]), (top_points[3]), (screen_pos[0] - w, screen_pos[1] + 2*h), (screen_pos[0], screen_pos[1] + 3*h)], side_color)
        pygame.gfxdraw.filled_polygon(self.screen, [(top_points[1]), (top_points[2]), (screen_pos[0], screen_pos[1] + 3*h), (screen_pos[0] + w, screen_pos[1] + 2*h)], side_color)
        
        # Draw top
        pygame.gfxdraw.aapolygon(self.screen, top_points, color)
        pygame.gfxdraw.filled_polygon(self.screen, top_points, color)


    def _render_ui(self):
        # Score
        score_text = self.font_main.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Gems
        gem_text = self.font_main.render(f"Gems: {self.gems_collected}/{self.NUM_GEMS}", True, self.COLOR_TEXT)
        self.screen.blit(gem_text, (10, 40))

        # Multiplier
        if self.score_multiplier > 1:
            mult_text = self.font_main.render(f"{self.score_multiplier}x", True, self.COLOR_MULTIPLIER)
            self.screen.blit(mult_text, (self.SCREEN_WIDTH - mult_text.get_width() - 10, 10))

        # Game Over / Win Message
        if self.game_over:
            msg = "YOU WIN!" if self.win else "GAME OVER"
            color = self.COLOR_PLAYER if self.win else self.COLOR_ENEMY
            end_text = self.font_main.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "gems_collected": self.gems_collected,
            "player_pos": tuple(self.player_pos),
        }

    def close(self):
        pygame.quit()
        
if __name__ == "__main__":
    # This block allows you to play the game manually
    # Make sure to unset the dummy video driver if you want to see the window
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Set up Pygame window for human play
    pygame.display.set_caption("Isometric Gem Collector")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    terminated = False
    truncated = False
    
    while running:
        movement = 0 # No-op
        space = 0
        shift = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                terminated = False
                truncated = False

        if not terminated and not truncated:
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
            obs, reward, terminated, truncated, info = env.step(action)
        
        # Convert observation back to a Pygame surface to display
        # The observation is (H, W, C), Pygame needs (W, H, C)
        # So we need to transpose it back
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30) # Run at 30 FPS

    env.close()