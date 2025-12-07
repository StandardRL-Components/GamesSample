import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press space to place a crystal."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Redirect laser beams through a crystal maze to reach the exits within the time limit."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.GRID_SIZE = 40
        self.GRID_W = self.WIDTH // self.GRID_SIZE
        self.GRID_H = self.HEIGHT // self.GRID_SIZE
        self.MAX_TIME = 60.0
        self.MAX_STEPS = int(self.MAX_TIME * self.FPS)
        self.MAX_CRYSTALS = 5

        # Colors
        self.COLOR_BG = (15, 20, 30)
        self.COLOR_WALL = (40, 50, 60)
        self.COLOR_GRID = (25, 30, 40)
        self.COLOR_CURSOR = (255, 255, 255, 100)
        self.COLOR_TEXT = (220, 220, 240)
        self.LASER_COLORS = {
            "red": ((255, 50, 50), (255, 150, 150)),
        }
        self.CRYSTAL_COLOR = (200, 200, 255)
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.ui_font = pygame.font.Font(None, 32)
        self.title_font = pygame.font.Font(None, 50)
        
        # Game state (initialized in reset)
        self.cursor_pos = None
        self.crystals = None
        self.walls = None
        self.lasers = None
        self.exits = None
        self.time_remaining = None
        self.last_space_held = None
        self.crystals_remaining = None
        self.score = None
        self.steps = None
        self.game_over = None
        self.game_won = None
        self.reward_this_step = 0
        
        # Initialize state variables
        # self.reset() is called by the test runner, no need to call it here.
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.cursor_pos = [self.GRID_W // 2, self.GRID_H // 2]
        self.crystals = []
        
        self._define_level()

        self.time_remaining = self.MAX_TIME
        self.crystals_remaining = self.MAX_CRYSTALS
        self.last_space_held = False
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        
        self._update_lasers()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.reward_this_step = 0
        self.steps += 1
        self.time_remaining -= 1.0 / self.FPS
        
        input_changed_lasers = self._handle_input(action)
        self._update_animations()
        
        if input_changed_lasers:
            self._update_lasers()
        
        self._apply_continuous_rewards()
        terminated = self._check_termination()
        reward = self.reward_this_step
        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _define_level(self):
        self.walls = []
        for x in range(self.GRID_W):
            self.walls.append(pygame.Rect(x, 0, 1, 1))
            self.walls.append(pygame.Rect(x, self.GRID_H - 1, 1, 1))
        for y in range(1, self.GRID_H - 1):
            self.walls.append(pygame.Rect(0, y, 1, 1))
            self.walls.append(pygame.Rect(self.GRID_W - 1, y, 1, 1))
            
        self.walls.append(pygame.Rect(5, 3, 1, 5))
        self.walls.append(pygame.Rect(10, 2, 1, 5))

        self.lasers = [
            {"origin": (2, 5), "dir": (1, 0), "color_key": "red", "was_solved": False}
        ]
        self.exits = [{"pos": (13, 7), "color_key": "red"}]
        
        for laser in self.lasers:
            laser['origin_world'] = self._grid_to_world(laser['origin'])
            laser['color'] = self.LASER_COLORS[laser['color_key']]
        for exit_obj in self.exits:
            exit_obj['color'] = self.LASER_COLORS[exit_obj['color_key']][0]

    def _handle_input(self, action):
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        
        if movement == 1: self.cursor_pos[1] -= 1
        elif movement == 2: self.cursor_pos[1] += 1
        elif movement == 3: self.cursor_pos[0] -= 1
        elif movement == 4: self.cursor_pos[0] += 1
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_W - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_H - 1)

        just_pressed_space = space_held and not self.last_space_held
        self.last_space_held = space_held
        
        if just_pressed_space and self.crystals_remaining > 0:
            is_occupied = any(tuple(self.cursor_pos) == tuple(c['pos']) for c in self.crystals) or \
                          any(w.collidepoint(self.cursor_pos) for w in self.walls) or \
                          any(tuple(l['origin']) == tuple(self.cursor_pos) for l in self.lasers) or \
                          any(tuple(e['pos']) == tuple(self.cursor_pos) for e in self.exits)

            if not is_occupied:
                self.crystals.append({'pos': list(self.cursor_pos), 'pulse': random.random() * math.pi * 2})
                self.crystals_remaining -= 1
                return True
        return False

    def _update_animations(self):
        for c in self.crystals:
            c['pulse'] = (c['pulse'] + 0.1) % (2 * math.pi)

    def _update_lasers(self):
        num_solved = 0
        
        for laser in self.lasers:
            laser['path'] = []
            laser['state'] = 'active'
            
            # Use grid coordinates for tracing
            current_grid_pos = list(laser['origin'])
            grid_dir = list(laser['dir'])
            
            laser['path'].append(self._grid_to_world(current_grid_pos))
            
            for _ in range(20): # Max 20 segments per laser
                
                # Find the next collision point by stepping one cell at a time
                temp_pos = list(current_grid_pos)
                collision_found = False
                while not collision_found:
                    temp_pos[0] += grid_dir[0]
                    temp_pos[1] += grid_dir[1]

                    # 1. Check for out of bounds
                    if not (0 <= temp_pos[0] < self.GRID_W and 0 <= temp_pos[1] < self.GRID_H):
                        laser['path'].append(self._grid_to_world(temp_pos))
                        laser['state'] = 'dead'
                        collision_found = True

                    # 2. Check for wall
                    elif any(w.collidepoint(temp_pos) for w in self.walls):
                        laser['path'].append(self._grid_to_world(temp_pos))
                        laser['state'] = 'dead'
                        self.reward_this_step -= 10
                        collision_found = True

                    # 3. Check for crystal
                    elif next((c for c in self.crystals if c['pos'] == temp_pos), None):
                        laser['path'].append(self._grid_to_world(temp_pos))
                        current_grid_pos = temp_pos
                        # 90-degree counter-clockwise rotation
                        grid_dir = [grid_dir[1], grid_dir[0]] if grid_dir[0] != 0 else [-grid_dir[1], 0]
                        collision_found = True

                    # 4. Check for exit
                    elif next((e for e in self.exits if e['pos'] == temp_pos and e['color_key'] == laser['color_key']), None):
                        laser['path'].append(self._grid_to_world(temp_pos))
                        laser['state'] = 'solved'
                        if not laser['was_solved']:
                           self.reward_this_step += 5
                           laser['was_solved'] = True
                        collision_found = True
                
                # If laser path ended, stop tracing segments for this laser
                if laser['state'] != 'active':
                    break
            
            if laser['state'] == 'solved':
                num_solved += 1
        
        if len(self.lasers) > 0 and num_solved == len(self.lasers):
            self.game_won = True
            self.reward_this_step += 50

    def _apply_continuous_rewards(self):
        for laser in self.lasers:
            if laser['state'] == 'solved':
                self.reward_this_step += 0.1
            elif laser['state'] == 'active':
                self.reward_this_step -= 0.01

    def _check_termination(self):
        if self.game_won:
            self.game_over = True
            return True
        # FIX: A dead laser is part of the puzzle, not a game-over condition.
        # if any(laser['state'] == 'dead' for laser in self.lasers):
        #     self.game_over = True
        #     return True
        if self.time_remaining <= 0 or self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True
        return False
        
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_grid()
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_grid(self):
        for x in range(0, self.WIDTH, self.GRID_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, self.GRID_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))

    def _render_game(self):
        for wall in self.walls:
            rect = pygame.Rect(wall.x * self.GRID_SIZE, wall.y * self.GRID_SIZE, self.GRID_SIZE, self.GRID_SIZE)
            pygame.draw.rect(self.screen, self.COLOR_WALL, rect)
            pygame.draw.rect(self.screen, self.COLOR_GRID, rect, 1)

        for exit_obj in self.exits:
            pos = self._grid_to_world(exit_obj['pos'])
            pygame.gfxdraw.filled_circle(self.screen, int(pos[0]), int(pos[1]), self.GRID_SIZE // 3, exit_obj['color'])
            pygame.gfxdraw.aacircle(self.screen, int(pos[0]), int(pos[1]), self.GRID_SIZE // 3, exit_obj['color'])

        for laser in self.lasers:
            if len(laser['path']) > 1:
                pygame.draw.aalines(self.screen, laser['color'][1], False, laser['path'], 5)
                pygame.draw.aalines(self.screen, laser['color'][0], False, laser['path'], 2)
                
                anim_phase = (self.steps * 4) % 40
                for i in range(len(laser['path']) - 1):
                    p1, p2 = laser['path'][i], laser['path'][i+1]
                    dist = math.hypot(p2[0]-p1[0], p2[1]-p1[1])
                    if dist == 0: continue
                    dx, dy = (p2[0]-p1[0])/dist, (p2[1]-p1[1])/dist
                    for d in range(int(anim_phase), int(dist), 40):
                        px, py = p1[0] + dx * d, p1[1] + dy * d
                        pygame.draw.circle(self.screen, laser['color'][1], (int(px), int(py)), 3)

        for crystal in self.crystals:
            center = self._grid_to_world(crystal['pos'])
            size = self.GRID_SIZE * 0.4
            points = [(center[0], center[1] - size), (center[0] + size, center[1]),
                      (center[0], center[1] + size), (center[0] - size, center[1])]
            
            pulse_radius = int(self.GRID_SIZE/2 * (1 + 0.1 * math.sin(crystal['pulse'])))
            glow_color = self.CRYSTAL_COLOR + (int(80 + 40 * math.sin(crystal['pulse'])),)
            temp_surf = pygame.Surface((pulse_radius*2, pulse_radius*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, glow_color, (pulse_radius, pulse_radius), pulse_radius)
            self.screen.blit(temp_surf, (int(center[0]-pulse_radius), int(center[1]-pulse_radius)))

            pygame.gfxdraw.filled_polygon(self.screen, [(int(p[0]), int(p[1])) for p in points], self.CRYSTAL_COLOR)
            pygame.gfxdraw.aapolygon(self.screen, [(int(p[0]), int(p[1])) for p in points], (255, 255, 255))
            
        if not self.game_over:
            cursor_rect = pygame.Rect(self.cursor_pos[0] * self.GRID_SIZE, self.cursor_pos[1] * self.GRID_SIZE, self.GRID_SIZE, self.GRID_SIZE)
            s = pygame.Surface((self.GRID_SIZE, self.GRID_SIZE), pygame.SRCALPHA)
            s.fill(self.COLOR_CURSOR)
            self.screen.blit(s, cursor_rect.topleft)
            pygame.draw.rect(self.screen, (255, 255, 255), cursor_rect, 1)

    def _render_ui(self):
        time_text = f"TIME: {max(0, self.time_remaining):.1f}"
        time_surf = self.ui_font.render(time_text, True, self.COLOR_TEXT)
        self.screen.blit(time_surf, (self.WIDTH - time_surf.get_width() - 10, 10))
        
        crystal_text = f"CRYSTALS: {self.crystals_remaining}"
        crystal_surf = self.ui_font.render(crystal_text, True, self.COLOR_TEXT)
        self.screen.blit(crystal_surf, (10, 10))

        if self.game_over:
            message = "LEVEL CLEARED" if self.game_won else "CONNECTION LOST"
            color = (100, 255, 100) if self.game_won else (255, 100, 100)
            msg_surf = self.title_font.render(message, True, color)
            msg_rect = msg_surf.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            
            bg_rect = msg_rect.inflate(20, 20)
            s = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
            s.fill((10, 10, 20, 200))
            self.screen.blit(s, bg_rect)
            pygame.draw.rect(self.screen, color, bg_rect, 1)
            self.screen.blit(msg_surf, msg_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining": self.time_remaining,
            "crystals_remaining": self.crystals_remaining,
        }

    def _grid_to_world(self, grid_pos):
        return [
            grid_pos[0] * self.GRID_SIZE + self.GRID_SIZE // 2,
            grid_pos[1] * self.GRID_SIZE + self.GRID_SIZE // 2
        ]

    def close(self):
        pygame.quit()