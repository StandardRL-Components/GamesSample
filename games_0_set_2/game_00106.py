import os
import os
import pygame

os.environ["SDL_VIDEODRIVER"] = "dummy"

import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move your character. Collect the gems and avoid the phantoms!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Collect shimmering gems in an isometric world while dodging cunning enemies to reach the target score before time runs out."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30

    # World and Grid
    GRID_WIDTH = 18
    GRID_HEIGHT = 12
    TILE_WIDTH_HALF = 24
    TILE_HEIGHT_HALF = 12
    WORLD_ORIGIN_X = SCREEN_WIDTH // 2
    WORLD_ORIGIN_Y = 100

    # Colors
    COLOR_BG = (20, 25, 40)
    COLOR_GRID = (40, 50, 70)
    
    COLOR_PLAYER = (255, 215, 0)
    COLOR_PLAYER_GLOW = (255, 215, 0, 50)
    
    GEM_COLORS = [(255, 0, 100), (0, 200, 255), (100, 255, 100)]
    
    COLOR_ENEMY = (120, 50, 200)
    COLOR_ENEMY_GLOW = (120, 50, 200, 50)
    
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_UI_TIMER_FG = (100, 255, 100)
    COLOR_UI_TIMER_BG = (255, 100, 100)
    
    # Game Parameters
    GAME_DURATION_SECONDS = 60
    WIN_GEM_COUNT = 20
    NUM_GEMS = 5
    NUM_ENEMIES = 3
    BASE_ENEMY_SPEED = 0.03 # grid units per step
    ENEMY_SPEED_INCREASE = 0.002
    PLAYER_INVULNERABILITY_FRAMES = 60 # 2 seconds

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
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 64)
        
        self.render_mode = render_mode
        self._max_episode_steps = self.GAME_DURATION_SECONDS * self.FPS

        self.grid_surface = self._create_grid_surface()
        
        # This will be called by the wrapper, but is useful for standalone validation
        # self.reset()
        # self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.gem_count = 0
        self.game_over = False
        self.win = False
        self.game_timer = self.GAME_DURATION_SECONDS * self.FPS

        self.player_pos = pygame.Vector2(self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2)
        self.player_visual_pos = self.player_pos.copy()
        self.player_invulnerable_timer = 0

        self.gems = []
        for _ in range(self.NUM_GEMS):
            self.gems.append(self._spawn_gem())
        
        self.enemies = [self._spawn_enemy() for _ in range(self.NUM_ENEMIES)]
        self._update_enemy_speed()
        
        self.particles = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement = action[0]
        
        reward = 0
        terminated = False

        # --- Update Timers ---
        self.game_timer -= 1
        if self.player_invulnerable_timer > 0:
            self.player_invulnerable_timer -= 1

        # --- Pre-move distance calculations for reward ---
        dist_gem_before = self._get_closest_distance(self.player_pos, self.gems)
        dist_enemy_before = self._get_closest_distance(self.player_pos, self.enemies)

        # --- Handle player movement ---
        if movement != 0 and self.player_visual_pos.distance_to(self.player_pos) < 0.1:
            target_pos = self.player_pos.copy()
            if movement == 1: target_pos.y -= 1  # Up
            elif movement == 2: target_pos.y += 1 # Down
            elif movement == 3: target_pos.x -= 1 # Left
            elif movement == 4: target_pos.x += 1 # Right
            
            if 0 <= target_pos.x < self.GRID_WIDTH and 0 <= target_pos.y < self.GRID_HEIGHT:
                self.player_pos = target_pos

        # --- Update entity visual positions (interpolation) ---
        self.player_visual_pos.move_towards_ip(self.player_pos, 0.3)
        for enemy in self.enemies:
            enemy['visual_pos'].move_towards_ip(enemy['pos'], 0.5)

        # --- Update enemy logic ---
        for enemy in self.enemies:
            if enemy['pos'].distance_to(enemy['path_end']) > 0.1:
                direction = (enemy['path_end'] - enemy['pos']).normalize()
                enemy['pos'] += direction * self.current_enemy_speed
            else:
                # Swap path points to patrol back and forth
                enemy['path_start'], enemy['path_end'] = enemy['path_end'], enemy['path_start']

        # --- Update particles ---
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1

        # --- Collision Detection ---
        # Player-Gem Collision
        for i, gem in enumerate(self.gems):
            if self.player_visual_pos.distance_to(gem['pos']) < 0.5:
                # sfx: gem_collect.wav
                reward += 10
                self.gem_count += 1
                self._create_particles(gem['pos'], gem['color'], 20)
                self.gems[i] = self._spawn_gem()
                self._update_enemy_speed()
                break
        
        # Player-Enemy Collision
        if self.player_invulnerable_timer == 0:
            for enemy in self.enemies:
                if self.player_visual_pos.distance_to(enemy['pos']) < 0.8:
                    # sfx: player_hit.wav
                    reward -= 5
                    self.player_invulnerable_timer = self.PLAYER_INVULNERABILITY_FRAMES
                    self._create_particles(self.player_pos, self.COLOR_ENEMY, 30)
                    # Reset player to center
                    self.player_pos = pygame.Vector2(self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2)
                    break

        # --- Post-move distance calculations for reward ---
        dist_gem_after = self._get_closest_distance(self.player_pos, self.gems)
        dist_enemy_after = self._get_closest_distance(self.player_pos, self.enemies)
        
        if dist_gem_before > dist_gem_after: reward += 1.0 # Moved closer to a gem
        if dist_enemy_before < dist_enemy_after: reward += 0.1 # Moved away from an enemy

        # --- Check Termination Conditions ---
        if self.gem_count >= self.WIN_GEM_COUNT:
            # sfx: game_win.wav
            reward += 100
            terminated = True
            self.win = True
            self.game_over = True
        
        if self.game_timer <= 0:
            # sfx: game_over.wav
            reward -= 100
            terminated = True
            self.win = False
            self.game_over = True

        self.steps += 1
        if self.steps >= self._max_episode_steps:
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
        self.screen.blit(self.grid_surface, (0, 0))
        
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "gem_count": self.gem_count,
            "timer_seconds": self.game_timer / self.FPS
        }

    # --- Rendering Methods ---

    def _render_game(self):
        # Create a list of all dynamic objects to sort for rendering
        renderables = []
        
        # Add player
        renderables.append({
            'type': 'player', 
            'pos': self.player_visual_pos, 
            'z': self.player_visual_pos.x + self.player_visual_pos.y
        })
        
        # Add enemies
        for enemy in self.enemies:
            renderables.append({
                'type': 'enemy', 
                'pos': enemy['visual_pos'], 
                'z': enemy['visual_pos'].x + enemy['visual_pos'].y
            })

        # Add gems
        for gem in self.gems:
            renderables.append({
                'type': 'gem', 
                'pos': gem['pos'], 
                'color': gem['color'], 
                'z': gem['pos'].x + gem['pos'].y
            })

        # Sort by z-index for correct isometric rendering
        renderables.sort(key=lambda r: r['z'])

        # Draw sorted objects
        for r in renderables:
            if r['type'] == 'player': self._draw_iso_diamond(self.player_visual_pos, self.COLOR_PLAYER, self.COLOR_PLAYER_GLOW, 1.0)
            elif r['type'] == 'enemy': self._draw_iso_diamond(r['pos'], self.COLOR_ENEMY, self.COLOR_ENEMY_GLOW, 0.9)
            elif r['type'] == 'gem': self._draw_iso_diamond(r['pos'], r['color'], None, 0.6, shimmer=True)

        # Draw particles on top of everything
        self._draw_particles()

    def _render_ui(self):
        # Gem count
        gem_text = self.font_small.render(f"GEMS: {self.gem_count} / {self.WIN_GEM_COUNT}", True, self.COLOR_UI_TEXT)
        self.screen.blit(gem_text, (10, 10))

        # Timer bar
        timer_pct = max(0, self.game_timer / (self.GAME_DURATION_SECONDS * self.FPS))
        bar_width = 200
        bar_height = 15
        bar_x = self.SCREEN_WIDTH - bar_width - 10
        bar_y = 10
        pygame.draw.rect(self.screen, self.COLOR_UI_TIMER_BG, (bar_x, bar_y, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_UI_TIMER_FG, (bar_x, bar_y, bar_width * timer_pct, bar_height))

        # Game Over / Win Text
        if self.game_over:
            text_str = "YOU WIN!" if self.win else "TIME UP!"
            color = self.COLOR_UI_TIMER_FG if self.win else self.COLOR_UI_TIMER_BG
            end_text = self.font_large.render(text_str, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)
            
    def _create_grid_surface(self):
        surface = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        for y in range(self.GRID_HEIGHT + 1):
            start_pos = self._iso_to_screen(pygame.Vector2(0, y))
            end_pos = self._iso_to_screen(pygame.Vector2(self.GRID_WIDTH, y))
            pygame.draw.line(surface, self.COLOR_GRID, start_pos, end_pos, 1)
        for x in range(self.GRID_WIDTH + 1):
            start_pos = self._iso_to_screen(pygame.Vector2(x, 0))
            end_pos = self._iso_to_screen(pygame.Vector2(x, self.GRID_HEIGHT))
            pygame.draw.line(surface, self.COLOR_GRID, start_pos, end_pos, 1)
        return surface

    def _draw_iso_diamond(self, pos, color, glow_color, size_mult=1.0, shimmer=False):
        screen_pos = self._iso_to_screen(pos)
        
        w = self.TILE_WIDTH_HALF * size_mult
        h = self.TILE_HEIGHT_HALF * size_mult
        
        if shimmer:
            shimmer_scale = 0.8 + 0.2 * math.sin(self.steps * 0.2 + pos.x)
            w *= shimmer_scale
            h *= shimmer_scale

        points = [
            (screen_pos.x, screen_pos.y - h),
            (screen_pos.x + w, screen_pos.y),
            (screen_pos.x, screen_pos.y + h),
            (screen_pos.x - w, screen_pos.y)
        ]
        
        if glow_color:
            gw = w * 2.0
            gh = h * 2.0
            glow_points = [
                (screen_pos.x, screen_pos.y - gh),
                (screen_pos.x + gw, screen_pos.y),
                (screen_pos.x, screen_pos.y + gh),
                (screen_pos.x - gw, screen_pos.y)
            ]
            pygame.gfxdraw.filled_polygon(self.screen, [(int(p[0]), int(p[1])) for p in glow_points], glow_color)
        
        if hasattr(self, 'player_invulnerable_timer') and self.player_invulnerable_timer > 0 and color == self.COLOR_PLAYER:
             if self.steps % 10 < 5: # Blink when invulnerable
                return

        pygame.gfxdraw.filled_polygon(self.screen, [(int(p[0]), int(p[1])) for p in points], color)
        pygame.gfxdraw.aapolygon(self.screen, [(int(p[0]), int(p[1])) for p in points], color)

    def _draw_particles(self):
        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            color = (*p['color'], alpha)
            temp_surf = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p['size'], p['size']), p['size'])
            self.screen.blit(temp_surf, (int(p['pos'].x - p['size']), int(p['pos'].y - p['size'])))

    # --- Helper Methods ---
    
    def _iso_to_screen(self, pos):
        screen_x = self.WORLD_ORIGIN_X + (pos.x - pos.y) * self.TILE_WIDTH_HALF
        screen_y = self.WORLD_ORIGIN_Y + (pos.x + pos.y) * self.TILE_HEIGHT_HALF
        return pygame.Vector2(screen_x, screen_y)

    def _spawn_gem(self):
        while True:
            pos = pygame.Vector2(
                self.np_random.integers(0, self.GRID_WIDTH),
                self.np_random.integers(0, self.GRID_HEIGHT)
            )
            # Ensure not spawning on player or other gems
            if pos.distance_to(self.player_pos) > 2 and all(pos.distance_to(g['pos']) > 1 for g in getattr(self, 'gems', [])):
                return {'pos': pos, 'color': self.np_random.choice(self.GEM_COLORS)}

    def _spawn_enemy(self):
        start = pygame.Vector2(
            self.np_random.integers(0, self.GRID_WIDTH),
            self.np_random.integers(0, self.GRID_HEIGHT)
        )
        end = pygame.Vector2(
            self.np_random.integers(0, self.GRID_WIDTH),
            self.np_random.integers(0, self.GRID_HEIGHT)
        )
        # Ensure path is not trivial
        while start.distance_to(end) < 5:
            end = pygame.Vector2(self.np_random.integers(0, self.GRID_WIDTH), self.np_random.integers(0, self.GRID_HEIGHT))

        return {
            'pos': start.copy(),
            'visual_pos': start.copy(),
            'path_start': start,
            'path_end': end
        }

    def _update_enemy_speed(self):
        self.current_enemy_speed = self.BASE_ENEMY_SPEED + (self.gem_count // 2) * self.ENEMY_SPEED_INCREASE

    def _get_closest_distance(self, pos, entities):
        if not entities:
            return float('inf')
        return min(pos.distance_to(e['pos']) for e in entities)

    def _create_particles(self, grid_pos, color, count):
        screen_pos = self._iso_to_screen(grid_pos)
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                'pos': screen_pos.copy(),
                'vel': pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed),
                'life': self.np_random.integers(15, 31),
                'max_life': 30,
                'color': color,
                'size': self.np_random.integers(2, 6)
            })

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space
        self.reset()
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
    # This block allows you to run the file directly to play the game
    # Note: Gymnasium environments are not typically run this way for training,
    # but it's useful for testing and visualization.
    
    # To play with a window, comment out the os.environ line at the top of the file
    play_in_window = True
    if "SDL_VIDEODRIVER" in os.environ and os.environ["SDL_VIDEODRIVER"] == "dummy":
        play_in_window = False

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup Pygame window for human play
    if play_in_window:
        pygame.display.set_caption(env.game_description)
        screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    terminated = False
    total_reward = 0.0
    
    while not terminated:
        # --- Human Controls ---
        movement = 0 # No-op
        space_held = 0
        shift_held = 0
        
        if play_in_window:
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
        else: # Simple agent for headless mode
            action = env.action_space.sample()

        action = [movement, space_held, shift_held]
        
        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # --- Rendering ---
        if play_in_window:
            # The observation is the rendered frame, so we just need to show it
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
        
        # --- Frame Rate ---
        env.clock.tick(env.FPS)

        if terminated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Gems: {info['gem_count']}")
            if play_in_window:
                # Wait a moment before closing
                pygame.time.wait(3000)

    env.close()