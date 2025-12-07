
# Generated: 2025-08-28T03:39:18.266117
# Source Brief: brief_02087.md
# Brief Index: 2087

        
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
        "Controls: Use arrow keys to move the cursor. Press Shift to cycle through crystal types. "
        "Press Space to place the selected crystal on the grid."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Redirect a laser beam through a crystalline cavern to hit the target. Strategically place "
        "refractive crystals (Blue: 90° turn, Yellow: 180° turn, Purple: Splitter) to guide the beam. "
        "You have a limited number of crystals and steps!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((640, 400))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 16)
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)
        
        # Game constants
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.GRID_W, self.GRID_H = 32, 20
        self.CELL_SIZE = 20
        self.MAX_STEPS = 600
        self.MAX_CRYSTALS = 3
        self.MAX_BOUNCES = 10
        
        # Colors
        self.COLOR_BG = (15, 10, 25)
        self.COLOR_GRID = (30, 20, 50)
        self.COLOR_LASER = (255, 80, 80)
        self.COLOR_TARGET = (80, 255, 80)
        self.COLOR_TARGET_GLOW = (20, 255, 20)
        self.COLOR_SOURCE = (80, 80, 255)
        self.COLOR_SOURCE_GLOW = (20, 20, 255)
        self.COLOR_CURSOR = (255, 255, 255, 100)
        self.CRYSTAL_COLORS = [
            (100, 150, 255),  # Blue: 90 deg right turn
            (255, 220, 100),  # Yellow: 180 deg turn
            (200, 100, 255),  # Purple: Splitter
        ]
        self.CRYSTAL_GLOWS = [
            (80, 120, 255),
            (255, 200, 80),
            (180, 80, 255),
        ]

        # State variables set in reset()
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.target_hit = False
        self.cursor_pos = [0, 0]
        self.selected_crystal_type = 0
        self.placed_crystals = []
        self.laser_source = (0, 0)
        self.laser_source_dir = (1, 0)
        self.target_pos = (0, 0)
        self.laser_path = []
        self.particles = []
        self.np_random = None
        
        # Initialize state variables
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.target_hit = False
        
        # Place entities, ensuring they are not on edges and are reasonably separated
        self.laser_source = (self.np_random.integers(1, 5), self.np_random.integers(1, self.GRID_H - 1))
        self.target_pos = (self.np_random.integers(self.GRID_W - 5, self.GRID_W - 1), self.np_random.integers(1, self.GRID_H - 1))
        while math.dist(self.laser_source, self.target_pos) < self.GRID_W / 2:
            self.target_pos = (self.np_random.integers(self.GRID_W - 5, self.GRID_W - 1), self.np_random.integers(1, self.GRID_H - 1))

        self.placed_crystals = []
        self.cursor_pos = [self.GRID_W // 2, self.GRID_H // 2]
        self.selected_crystal_type = 0
        self.particles = []
        
        self._recalculate_laser()

        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0.0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_press = action[1] == 1  # Boolean
        shift_press = action[2] == 1  # Boolean
        
        reward = -0.1  # Per-step penalty

        # 1. Handle player actions
        if movement == 1: self.cursor_pos[1] -= 1  # Up
        elif movement == 2: self.cursor_pos[1] += 1  # Down
        elif movement == 3: self.cursor_pos[0] -= 1  # Left
        elif movement == 4: self.cursor_pos[0] += 1  # Right
        
        self.cursor_pos[0] %= self.GRID_W
        self.cursor_pos[1] %= self.GRID_H

        if shift_press:
            self.selected_crystal_type = (self.selected_crystal_type + 1) % len(self.CRYSTAL_COLORS)
        
        if space_press:
            if self._place_crystal():
                reward -= 1.0  # Cost for placing a crystal
                bounce_reward, hit_target = self._recalculate_laser()
                reward += bounce_reward
                if hit_target and not self.target_hit:
                    self.target_hit = True
                    reward += 100.0
        
        # 2. Update state and check for termination
        self.steps += 1
        terminated = False
        if self.target_hit:
            terminated = True
            self.game_over = True
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
            if not self.target_hit:
                reward -= 100.0  # Failure penalty on timeout

        self.score += reward
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _place_crystal(self):
        if len(self.placed_crystals) >= self.MAX_CRYSTALS:
            return False # Max crystals reached
        
        pos_tuple = tuple(self.cursor_pos)
        if pos_tuple == self.laser_source or pos_tuple == self.target_pos:
            return False # Cannot place on source or target
        
        if any(c['pos'] == pos_tuple for c in self.placed_crystals):
            return False # Cannot place on another crystal
            
        self.placed_crystals.append({'pos': pos_tuple, 'type': self.selected_crystal_type})
        # sfx: crystal_place.wav
        return True

    def _recalculate_laser(self):
        self.laser_path = []
        bounce_reward = 0.0
        target_is_hit = False
        
        beams = [{'pos': self.laser_source, 'dir': self.laser_source_dir, 'bounces': self.MAX_BOUNCES}]
        visited_beam_states = set()
        max_iterations = self.MAX_BOUNCES * (self.GRID_W + self.GRID_H)

        while beams and max_iterations > 0:
            max_iterations -= 1
            beam = beams.pop(0)
            pos, direction, bounces = beam['pos'], beam['dir'], beam['bounces']
            
            state_tuple = (pos, direction, bounces)
            if state_tuple in visited_beam_states: continue
            visited_beam_states.add(state_tuple)

            if bounces <= 0: continue

            start_pos = pos
            current_pos = start_pos
            
            for _ in range(max(self.GRID_W, self.GRID_H)):
                next_pos = (current_pos[0] + direction[0], current_pos[1] + direction[1])
                
                if not (0 <= next_pos[0] < self.GRID_W and 0 <= next_pos[1] < self.GRID_H):
                    self.laser_path.append({'start': start_pos, 'end': current_pos, 'color': self.COLOR_LASER})
                    self._create_particles(self._grid_to_pixel_center(current_pos), self.COLOR_LASER)
                    # sfx: laser_bounce_wall.wav
                    bounce_reward += 5.0
                    new_dir = direction
                    if not (0 <= next_pos[0] < self.GRID_W): new_dir = (-direction[0], direction[1])
                    if not (0 <= next_pos[1] < self.GRID_H): new_dir = (direction[0], -direction[1])
                    beams.append({'pos': current_pos, 'dir': new_dir, 'bounces': bounces - 1})
                    break

                if next_pos == self.target_pos:
                    self.laser_path.append({'start': start_pos, 'end': next_pos, 'color': self.COLOR_TARGET})
                    self._create_particles(self._grid_to_pixel_center(next_pos), self.COLOR_TARGET, 20)
                    # sfx: target_hit.wav
                    target_is_hit = True
                    break

                crystal_hit = next((c for c in self.placed_crystals if c['pos'] == next_pos), None)
                if crystal_hit:
                    self.laser_path.append({'start': start_pos, 'end': next_pos, 'color': self.COLOR_LASER})
                    self._create_particles(self._grid_to_pixel_center(next_pos), self.CRYSTAL_COLORS[crystal_hit['type']])
                    # sfx: laser_bounce_crystal.wav
                    bounce_reward += 5.0
                    ctype, dx, dy = crystal_hit['type'], direction[0], direction[1]
                    
                    if ctype == 0: # Blue: 90 deg right turn
                        beams.append({'pos': next_pos, 'dir': (dy, -dx), 'bounces': bounces - 1})
                    elif ctype == 1: # Yellow: 180 deg turn
                        beams.append({'pos': next_pos, 'dir': (-dx, -dy), 'bounces': bounces - 1})
                    elif ctype == 2: # Purple: Splitter
                        beams.append({'pos': next_pos, 'dir': (dy, -dx), 'bounces': bounces - 1})
                        beams.append({'pos': next_pos, 'dir': (-dy, dx), 'bounces': bounces - 1})
                    break
                current_pos = next_pos
            else:
                self.laser_path.append({'start': start_pos, 'end': current_pos, 'color': self.COLOR_LASER})
                
        return bounce_reward, target_is_hit

    def _grid_to_pixel_center(self, grid_pos):
        x = grid_pos[0] * self.CELL_SIZE + self.CELL_SIZE // 2
        y = grid_pos[1] * self.CELL_SIZE + self.CELL_SIZE // 2
        return x, y

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def _render_game(self):
        for x in range(0, self.SCREEN_WIDTH, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))

        self._update_and_draw_particles()

        for segment in self.laser_path:
            start_px = self._grid_to_pixel_center(segment['start'])
            end_px = self._grid_to_pixel_center(segment['end'])
            color = segment['color']
            pygame.draw.line(self.screen, (min(255,color[0]+50), min(255,color[1]+50), min(255,color[2]+50)), start_px, end_px, 5)
            pygame.draw.line(self.screen, (255,255,255), start_px, end_px, 1)

        self._draw_glowing_circle(self.screen, self.COLOR_SOURCE, self.COLOR_SOURCE_GLOW, self._grid_to_pixel_center(self.laser_source), 8)
        self._draw_glowing_circle(self.screen, self.COLOR_TARGET, self.COLOR_TARGET_GLOW, self._grid_to_pixel_center(self.target_pos), 8)
        
        for crystal in self.placed_crystals:
            self._draw_crystal(crystal['pos'], crystal['type'])

        cursor_rect = pygame.Rect(self.cursor_pos[0] * self.CELL_SIZE, self.cursor_pos[1] * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
        cursor_surface = pygame.Surface(cursor_rect.size, pygame.SRCALPHA)
        cursor_surface.fill(self.COLOR_CURSOR)
        self.screen.blit(cursor_surface, cursor_rect.topleft)

    def _draw_crystal(self, pos, type):
        center_px = self._grid_to_pixel_center(pos)
        color = self.CRYSTAL_COLORS[type]
        glow_color = self.CRYSTAL_GLOWS[type]
        size = self.CELL_SIZE * 0.4
        
        points = [
            (center_px[0], center_px[1] - size), (center_px[0] + size, center_px[1]),
            (center_px[0], center_px[1] + size), (center_px[0] - size, center_px[1]),
        ]
        
        pygame.gfxdraw.filled_polygon(self.screen, points, (*glow_color, 60))
        pygame.gfxdraw.aapolygon(self.screen, points, (*glow_color, 120))
        pygame.gfxdraw.filled_polygon(self.screen, points, color)
        pygame.gfxdraw.aapolygon(self.screen, points, (255,255,255,200))

    def _draw_glowing_circle(self, surface, color, glow_color, pos, radius):
        pos = (int(pos[0]), int(pos[1]))
        pygame.gfxdraw.filled_circle(surface, pos[0], pos[1], int(radius * 1.5), (*glow_color, 60))
        pygame.gfxdraw.aacircle(surface, pos[0], pos[1], int(radius * 1.5), (*glow_color, 120))
        pygame.gfxdraw.filled_circle(surface, pos[0], pos[1], radius, color)
        pygame.gfxdraw.aacircle(surface, pos[0], pos[1], radius, (255, 255, 255, 200))
        
    def _render_ui(self):
        panel_rect = pygame.Rect(0, 0, self.SCREEN_WIDTH, 30)
        pygame.draw.rect(self.screen, (0,0,0,150), panel_rect)

        steps_text = f"Steps: {self.steps}/{self.MAX_STEPS}"
        steps_surf = self.font_small.render(steps_text, True, (255, 255, 255))
        self.screen.blit(steps_surf, (10, 7))

        score_text = f"Score: {self.score:.1f}"
        score_surf = self.font_small.render(score_text, True, (255, 255, 255))
        self.screen.blit(score_surf, (self.SCREEN_WIDTH - score_surf.get_width() - 10, 7))
        
        crystal_text = self.font_small.render("Selected:", True, (255, 255, 255))
        self.screen.blit(crystal_text, (self.SCREEN_WIDTH // 2 - 80, 7))
        self._draw_crystal((self.SCREEN_WIDTH // 2 // self.CELL_SIZE, 15 // self.CELL_SIZE), self.selected_crystal_type)

        if self.game_over:
            msg = "TARGET HIT!" if self.target_hit else "TIME UP!"
            color = self.COLOR_TARGET if self.target_hit else self.COLOR_LASER
            msg_surf = self.font_large.render(msg, True, color)
            msg_rect = msg_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            pygame.draw.rect(self.screen, (0,0,0,180), msg_rect.inflate(20, 20))
            self.screen.blit(msg_surf, msg_rect)

    def _create_particles(self, pos, color, count=10, life=15, speed=2):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            vel = [math.cos(angle) * speed * self.np_random.uniform(0.5, 1.5),
                   math.sin(angle) * speed * self.np_random.uniform(0.5, 1.5)]
            self.particles.append({
                'pos': list(pos), 'vel': vel, 'life': life,
                'max_life': life, 'color': color, 'radius': self.np_random.uniform(2, 5)
            })

    def _update_and_draw_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)
            else:
                alpha = int(255 * (p['life'] / p['max_life']))
                radius = int(p['radius'] * (p['life'] / p['max_life']))
                if radius > 0:
                    pos = (int(p['pos'][0]), int(p['pos'][1]))
                    pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, (*p['color'], alpha))

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    try:
        screen = pygame.display.set_mode((640, 400))
        pygame.display.set_caption("Laser Redirect")
    except pygame.error:
        print("Pygame display could not be initialized. Running in headless mode.")
        screen = None

    clock = pygame.time.Clock()

    while not done:
        action = [0, 0, 0] # Default no-op
        should_step = False
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                should_step = True
                if event.key == pygame.K_UP: action[0] = 1
                elif event.key == pygame.K_DOWN: action[0] = 2
                elif event.key == pygame.K_LEFT: action[0] = 3
                elif event.key == pygame.K_RIGHT: action[0] = 4
                elif event.key == pygame.K_SPACE: action[1] = 1
                elif event.key in [pygame.K_LSHIFT, pygame.K_RSHIFT]: action[2] = 1

        if should_step:
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Step: {info['steps']}, Score: {info['score']:.2f}, Reward: {reward:.2f}, Terminated: {terminated}")
            if terminated:
                if screen:
                    frame = np.transpose(obs, (1, 0, 2))
                    surf = pygame.surfarray.make_surface(frame)
                    screen.blit(surf, (0, 0))
                    pygame.display.flip()
                    pygame.time.wait(2000)
                obs, info = env.reset()

        if done:
            break
            
        if screen:
            frame = np.transpose(obs, (1, 0, 2))
            surf = pygame.surfarray.make_surface(frame)
            screen.blit(surf, (0, 0))
            pygame.display.flip()
        
        clock.tick(30)

    env.close()