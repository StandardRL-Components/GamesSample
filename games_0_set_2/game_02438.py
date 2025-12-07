
# Generated: 2025-08-27T20:22:46.959956
# Source Brief: brief_02438.md
# Brief Index: 2438

        
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
        "Controls: Arrow keys to move the cursor. Press Space to place a reflector crystal."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "An isometric puzzle game. Place reflector crystals to bend a light beam and illuminate all target crystals before the timer runs out."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    GAME_DURATION_SECONDS = 60
    MAX_STEPS = GAME_DURATION_SECONDS * FPS

    GRID_W, GRID_H = 22, 14
    TILE_W, TILE_H = 32, 16
    TILE_W_HALF, TILE_H_HALF = TILE_W // 2, TILE_H // 2
    
    ORIGIN_X = WIDTH // 2
    ORIGIN_Y = 80

    COLOR_BG = (20, 25, 40)
    COLOR_WALL = (50, 60, 80)
    COLOR_WALL_OUTLINE = (70, 80, 100)
    COLOR_TARGET_UNLIT = (40, 80, 150)
    COLOR_TARGET_LIT = (100, 220, 255)
    COLOR_PLACED_CRYSTAL = (255, 255, 255)
    COLOR_LIGHT_BEAM = (255, 255, 100)
    COLOR_TEXT = (220, 220, 220)
    COLOR_TIMER_WARN = (255, 100, 100)
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 28)
        self.font_game_over = pygame.font.Font(None, 64)
        
        self.cursor_pos = [0, 0]
        self.placed_crystals = []
        self.target_crystals = []
        self.light_source = {}
        self.cavern_walls = []
        self.crystals_remaining = 0
        self.time_left = 0
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.prev_space_held = False
        self.particles = []
        self.light_path = []
        self.lit_targets = set()
        
        self.reset()
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            # Use python's random for level generation for simplicity
            random.seed(seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.time_left = self.MAX_STEPS
        
        self._generate_level()
        
        self.cursor_pos = [self.GRID_W // 2, self.GRID_H // 2]
        self.placed_crystals = []
        self.prev_space_held = False
        self.particles = []
        
        self._update_light_path()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0
        
        self._move_cursor(movement)

        space_pressed = space_held and not self.prev_space_held
        if space_pressed:
            placement_reward = self._place_crystal()
            reward += placement_reward
        self.prev_space_held = space_held

        self.steps += 1
        self.time_left -= 1
        self._update_particles()
        
        # Continuous reward for lit targets
        reward += len(self.lit_targets) * 0.005
        self.score += reward

        terminated = self._check_termination()
        if terminated:
            if self.win:
                reward += 100
                self.score += 100
            else:
                reward -= 100
                self.score -= 100
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _generate_level(self):
        self.light_source = {'pos': (0, self.GRID_H // 2), 'dir': (1, 0)}
        self.target_crystals = []
        
        # Ensure some targets are reachable
        for i in range(3):
            x = random.randint(self.GRID_W - 5, self.GRID_W - 2)
            y = random.randint(2, self.GRID_H - 3)
            if (x, y) not in self.target_crystals:
                 self.target_crystals.append((x, y))

        for i in range(2):
            x = random.randint(3, self.GRID_W - 6)
            y = random.randint(2, self.GRID_H - 3)
            if (x, y) not in self.target_crystals:
                 self.target_crystals.append((x, y))

        self.cavern_walls = []
        for i in range(15):
             x = random.randint(1, self.GRID_W - 2)
             y = random.randint(1, self.GRID_H - 2)
             if (x,y) != self.light_source['pos'] and (x,y) not in self.target_crystals:
                 self.cavern_walls.append((x,y))

        self.crystals_remaining = 8

    def _move_cursor(self, movement):
        x, y = self.cursor_pos
        if movement == 1: y -= 1  # Up
        elif movement == 2: y += 1  # Down
        elif movement == 3: x -= 1  # Left
        elif movement == 4: x += 1  # Right
        
        self.cursor_pos[0] = np.clip(x, 0, self.GRID_W - 1)
        self.cursor_pos[1] = np.clip(y, 0, self.GRID_H - 1)

    def _place_crystal(self):
        pos = tuple(self.cursor_pos)
        is_valid_spot = (pos not in [c['pos'] for c in self.placed_crystals] and
                         pos not in self.target_crystals and
                         pos != self.light_source['pos'] and
                         pos not in self.cavern_walls)

        if self.crystals_remaining > 0 and is_valid_spot:
            self.placed_crystals.append({'pos': pos, 'anim': 0.0})
            self.crystals_remaining -= 1
            
            # sfx: place_crystal.wav
            self._spawn_particles(pos, 15, self.COLOR_PLACED_CRYSTAL)
            
            prev_lit_count = len(self.lit_targets)
            self._update_light_path()
            newly_lit_count = len(self.lit_targets) - prev_lit_count
            
            # Event reward for new lit targets, penalty for placing
            return -0.01 + (newly_lit_count * 5.0)
        return 0

    def _update_light_path(self):
        self.light_path = []
        self.lit_targets = set()
        
        reflectors = {c['pos'] for c in self.placed_crystals}
        targets = {pos: i for i, pos in enumerate(self.target_crystals)}
        walls = set(self.cavern_walls)

        active_rays = [(self.light_source['pos'], self.light_source['dir'])]
        processed_reflections = set()
        
        ray_limit = 50 # Prevent infinite loops
        while active_rays and ray_limit > 0:
            ray_limit -= 1
            pos, direction = active_rays.pop(0)
            
            current_pos = list(pos)
            path_segment = [tuple(current_pos)]
            
            for _ in range(self.GRID_W + self.GRID_H):
                current_pos[0] += direction[0]
                current_pos[1] += direction[1]
                t_pos = tuple(current_pos)

                if not (0 <= t_pos[0] < self.GRID_W and 0 <= t_pos[1] < self.GRID_H):
                    break
                
                path_segment.append(t_pos)

                if t_pos in targets:
                    self.lit_targets.add(targets[t_pos])
                    # sfx: target_activated.wav
                
                if t_pos in walls:
                    # sfx: light_absorb.wav
                    break

                if t_pos in reflectors:
                    if (t_pos, direction) not in processed_reflections:
                        # sfx: light_reflect.wav
                        new_direction = (-direction[1], direction[0]) # 90-degree rotation
                        active_rays.append((t_pos, new_direction))
                        processed_reflections.add((t_pos, direction))
                    break
            
            self.light_path.append(path_segment)

    def _check_termination(self):
        if len(self.lit_targets) == len(self.target_crystals):
            self.win = True
            return True
        if self.time_left <= 0:
            return True
        if self.crystals_remaining == 0:
            # Check if a solution is still possible. If not, end early.
            # For now, we only end if all crystals are placed and it's not a win.
            # A more complex check could see if light paths have stabilized.
            all_placed_and_not_win = True
            return all_placed_and_not_win
        return self.steps >= self.MAX_STEPS

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
            "time_left": self.time_left,
            "crystals_remaining": self.crystals_remaining,
            "lit_targets": len(self.lit_targets),
        }

    def _iso_to_screen(self, x, y):
        screen_x = self.ORIGIN_X + (x - y) * self.TILE_W_HALF
        screen_y = self.ORIGIN_Y + (x + y) * self.TILE_H_HALF
        return int(screen_x), int(screen_y)

    def _render_game(self):
        # Draw grid floors for reference (optional, subtle)
        for y in range(self.GRID_H):
            for x in range(self.GRID_W):
                sx, sy = self._iso_to_screen(x, y)
                points = [
                    self._iso_to_screen(x, y),
                    self._iso_to_screen(x + 1, y),
                    self._iso_to_screen(x + 1, y + 1),
                    self._iso_to_screen(x, y + 1)
                ]
                pygame.gfxdraw.filled_polygon(self.screen, points, (30,35,50))

        # Draw walls
        for x, y in self.cavern_walls:
            self._draw_iso_cube(x, y, self.COLOR_WALL, self.COLOR_WALL_OUTLINE)

        # Draw target crystals
        for i, (x, y) in enumerate(self.target_crystals):
            is_lit = i in self.lit_targets
            color = self.COLOR_TARGET_LIT if is_lit else self.COLOR_TARGET_UNLIT
            self._draw_iso_crystal(x, y, color, is_lit)

        # Draw placed crystals
        for crystal in self.placed_crystals:
            x, y = crystal['pos']
            if crystal['anim'] < 1.0: crystal['anim'] += 0.1
            self._draw_iso_crystal(x, y, self.COLOR_PLACED_CRYSTAL, True, scale=crystal['anim'])

        # Draw light source
        sx, sy = self._iso_to_screen(self.light_source['pos'][0], self.light_source['pos'][1])
        pygame.gfxdraw.filled_circle(self.screen, sx, sy, 5, self.COLOR_LIGHT_BEAM)
        pygame.gfxdraw.aacircle(self.screen, sx, sy, 5, self.COLOR_LIGHT_BEAM)

        # Draw light paths
        for segment in self.light_path:
            if len(segment) > 1:
                points = [self._iso_to_screen(x, y) for x, y in segment]
                pygame.draw.aalines(self.screen, self.COLOR_LIGHT_BEAM, False, points, 2)

        # Draw particles
        for p in self.particles:
            sx, sy = self._iso_to_screen(p['pos'][0], p['pos'][1])
            life_alpha = int(255 * (p['life'] / p['max_life']))
            color = (*p['color'], life_alpha)
            radius = int(p['size'] * (p['life'] / p['max_life']))
            if radius > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(sx + p['x']), int(sy + p['y']), radius, color)

        # Draw cursor
        if not self.game_over:
            cx, cy = self.cursor_pos
            sx, sy = self._iso_to_screen(cx, cy)
            pulse = (math.sin(self.steps * 0.2) + 1) / 2
            color = (255, 255, 255, 100 + pulse * 100)
            
            points = [
                self._iso_to_screen(cx, cy),
                self._iso_to_screen(cx + 1, cy),
                self._iso_to_screen(cx + 1, cy + 1),
                self._iso_to_screen(cx, cy + 1)
            ]
            pygame.gfxdraw.aapolygon(self.screen, points, color)
            pygame.gfxdraw.filled_polygon(self.screen, points, (color[0], color[1], color[2], color[3]//4))

    def _draw_iso_cube(self, x, y, color, outline_color):
        sx, sy = self._iso_to_screen(x, y)
        top_points = [
            (sx, sy),
            (sx + self.TILE_W_HALF, sy + self.TILE_H_HALF),
            (sx, sy + self.TILE_H),
            (sx - self.TILE_W_HALF, sy + self.TILE_H_HALF)
        ]
        pygame.gfxdraw.filled_polygon(self.screen, top_points, color)
        pygame.gfxdraw.aapolygon(self.screen, top_points, outline_color)

    def _draw_iso_crystal(self, x, y, color, is_lit, scale=1.0):
        sx, sy = self._iso_to_screen(x, y)
        
        h = self.TILE_H * scale
        w = self.TILE_W_HALF * scale

        points = [
            (sx, sy - h * 0.5),
            (sx + w * 0.7, sy),
            (sx, sy + h * 0.5),
            (sx - w * 0.7, sy)
        ]
        
        if is_lit:
            # Glow effect
            for i in range(4, 0, -1):
                glow_color = (*color, 30 // i)
                pygame.gfxdraw.filled_polygon(self.screen, points, glow_color)
                points = [(p[0] * 1.1 - sx * 0.1, p[1] * 1.1 - sy * 0.1) for p in points]
            # Reset points for main shape
            points = [
                (sx, sy - h * 0.5),
                (sx + w * 0.7, sy),
                (sx, sy + h * 0.5),
                (sx - w * 0.7, sy)
            ]
        
        pygame.gfxdraw.filled_polygon(self.screen, points, color)
        pygame.gfxdraw.aapolygon(self.screen, points, (255, 255, 255, 180))

    def _render_ui(self):
        # Crystals remaining
        crystal_text = f"REFLECTORS: {self.crystals_remaining}"
        text_surf = self.font_ui.render(crystal_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surf, (10, 10))
        
        # Targets lit
        target_text = f"TARGETS: {len(self.lit_targets)}/{len(self.target_crystals)}"
        text_surf = self.font_ui.render(target_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surf, (10, 35))

        # Timer
        seconds_left = self.time_left / self.FPS
        timer_text = f"TIME: {seconds_left:.1f}"
        timer_color = self.COLOR_TIMER_WARN if seconds_left < 10 else self.COLOR_TEXT
        text_surf = self.font_ui.render(timer_text, True, timer_color)
        text_rect = text_surf.get_rect(topright=(self.WIDTH - 10, 10))
        self.screen.blit(text_surf, text_rect)

        # Game Over message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            msg = "SYSTEM ONLINE" if self.win else "SYSTEM FAILURE"
            color = self.COLOR_TARGET_LIT if self.win else self.COLOR_TIMER_WARN
            text_surf = self.font_game_over.render(msg, True, color)
            text_rect = text_surf.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            self.screen.blit(text_surf, text_rect)

    def _spawn_particles(self, pos, count, color):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            self.particles.append({
                'pos': pos,
                'x': 0, 'y': 0,
                'vx': math.cos(angle) * speed,
                'vy': math.sin(angle) * speed,
                'life': 20, 'max_life': 20,
                'size': random.uniform(2, 5),
                'color': color
            })

    def _update_particles(self):
        for p in self.particles:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc is False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Isometric Light Puzzle")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    print(env.user_guide)

    while running:
        movement = 0 # No-op
        space_held = 0
        shift_held = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
        
        action = [movement, space_held, shift_held]
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Transpose observation back for Pygame display
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']:.2f}")
            pygame.time.wait(3000)
            obs, info = env.reset()
            total_reward = 0
            
        clock.tick(env.FPS)
        
    env.close()