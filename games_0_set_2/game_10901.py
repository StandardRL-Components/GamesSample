import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T11:10:20.145769
# Source Brief: brief_00901.md
# Brief Index: 901
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class Particle:
    def __init__(self, pos, vel, life, start_color, end_color, start_radius, end_radius=0):
        self.pos = list(pos)
        self.vel = list(vel)
        self.life = life
        self.max_life = life
        self.start_color = start_color
        self.end_color = end_color
        self.start_radius = start_radius
        self.end_radius = end_radius

    def update(self):
        self.pos[0] += self.vel[0]
        self.pos[1] += self.vel[1]
        self.vel[1] += 0.05 # gravity
        self.life -= 1
        return self.life > 0

    def draw(self, surface):
        if self.life <= 0:
            return
        
        progress = self.life / self.max_life
        
        current_color = [
            int(self.end_color[i] + (self.start_color[i] - self.end_color[i]) * progress)
            for i in range(3)
        ]
        current_radius = int(self.end_radius + (self.start_radius - self.end_radius) * progress)

        if current_radius > 0:
            pygame.gfxdraw.filled_circle(surface, int(self.pos[0]), int(self.pos[1]), current_radius, current_color)


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "A puzzle game where you place and guide clones using portals and switches to reach the exit."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move the cursor. Press space to create a clone and shift to place or clear portals."
    )
    auto_advance = False

    # --- CONSTANTS ---
    WIDTH, HEIGHT = 640, 400
    GRID_W, GRID_H = 32, 20
    CELL_SIZE = 20
    MAX_STEPS = 2000

    # Colors
    COLOR_BG = (15, 20, 40)
    COLOR_GRID = (30, 40, 60)
    COLOR_WALL = (70, 80, 110)
    COLOR_TEXT = (220, 220, 220)
    COLOR_CURSOR = (255, 255, 100)
    COLOR_CLONE = (0, 255, 150)
    COLOR_PORTAL_IN = (50, 150, 255)
    COLOR_PORTAL_OUT = (255, 150, 50)
    COLOR_HAZARD = (255, 50, 50)
    COLOR_SWITCH_OFF = (200, 100, 100)
    COLOR_SWITCH_ON = (100, 200, 100)
    COLOR_DOOR_CLOSED = (180, 180, 180)
    COLOR_DOOR_OPEN = (60, 60, 60)
    COLOR_EXIT = (200, 0, 255)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Arial", 16)
        self.font_large = pygame.font.SysFont("Arial", 24)
        
        self.render_mode = render_mode
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.cursor_pos = [0, 0]
        self.clones = []
        self.portals = {'entry': None, 'exit': None}
        self.portal_placement_mode = 'entry'
        self.particles = []
        self.level_data = np.zeros((self.GRID_W, self.GRID_H), dtype=int)
        self.exit_pos = None
        self.switches = {}
        self.doors = {}
        self.hazards = []
        self.step_rewards = []
        
        self.prev_space_held = False
        self.prev_shift_held = False

        self.level_num = 0
        self.combo = 0
        self.combo_timer = 0
        
        # self.reset() # reset is called by the wrapper
        # self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.step_rewards = []
        
        if options and 'level' in options:
            self.level_num = options['level']
        else:
            self.level_num = 1

        self._generate_level(self.level_num)

        self.cursor_pos = [self.GRID_W // 2, self.GRID_H // 2]
        self.clones = []
        self.portals = {'entry': None, 'exit': None}
        self.portal_placement_mode = 'entry'
        self.particles = []
        
        self.prev_space_held = False
        self.prev_shift_held = False

        self.combo = 0
        self.combo_timer = 0
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        self.step_rewards = [-0.1] # Small penalty for taking a step

        # --- Handle Actions ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        self._handle_movement(movement)
        
        if space_held and not self.prev_space_held:
            self._create_clone() # sfx: clone_create
        
        if shift_held and not self.prev_shift_held:
            self._place_portal() # sfx: portal_place

        self.prev_space_held, self.prev_shift_held = space_held, shift_held
        
        # --- Update Game State ---
        self._update_clones()
        self._update_particles()
        self._update_switches_and_doors()
        self._update_combo()

        # --- Calculate Step Result ---
        reward = self._calculate_reward()
        self.score += reward
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        if truncated:
            self.game_over = True
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _generate_level(self, level_num):
        self.level_data.fill(0)
        self.switches.clear()
        self.doors.clear()
        self.hazards.clear()
        
        # Add border walls
        self.level_data[0, :] = 1
        self.level_data[-1, :] = 1
        self.level_data[:, 0] = 1
        self.level_data[:, -1] = 1

        # Level designs
        if level_num == 1: # Simple path
            self.exit_pos = (self.GRID_W - 3, self.GRID_H // 2)
        elif level_num == 2: # One switch
            self.exit_pos = (self.GRID_W - 3, self.GRID_H // 2)
            self.level_data[self.GRID_W - 6, 5:self.GRID_H-5] = 1
            switch_pos = (10, self.GRID_H // 2)
            door_pos = (self.GRID_W - 6, self.GRID_H // 2)
            self.switches[switch_pos] = {'activated': False, 'door_pos': door_pos}
            self.doors[door_pos] = {'open': False}
        elif level_num == 3: # Portal required
            self.exit_pos = (self.GRID_W - 3, 3)
            self.level_data[5:self.GRID_W - 5, 8] = 1
            self.level_data[5:self.GRID_W - 5, self.GRID_H - 8] = 1
            switch_pos = (self.GRID_W // 2, self.GRID_H - 4)
            door_pos = (self.GRID_W // 2, 8)
            self.switches[switch_pos] = {'activated': False, 'door_pos': door_pos}
            self.doors[door_pos] = {'open': False}
        else: # More complex level with hazards
            self.level_num = 4
            self.exit_pos = (self.GRID_W - 3, self.GRID_H // 2)
            self.level_data[10, 5:self.GRID_H-5] = 1
            self.level_data[20, 5:self.GRID_H-5] = 1
            for y in range(5, self.GRID_H - 5, 2):
                self.hazards.append((15, y))
            switch_pos = (5, 5)
            door_pos = (10, self.GRID_H // 2)
            self.switches[switch_pos] = {'activated': False, 'door_pos': door_pos}
            self.doors[door_pos] = {'open': False}

    def _handle_movement(self, movement):
        dx, dy = 0, 0
        if movement == 1: dy = -1
        elif movement == 2: dy = 1
        elif movement == 3: dx = -1
        elif movement == 4: dx = 1
        
        if dx != 0 or dy != 0:
            new_x = self.cursor_pos[0] + dx
            new_y = self.cursor_pos[1] + dy
            if 0 <= new_x < self.GRID_W and 0 <= new_y < self.GRID_H:
                self.cursor_pos = [new_x, new_y]

    def _create_clone(self):
        pos = tuple(self.cursor_pos)
        if self.level_data[pos] == 1 or pos == self.exit_pos:
            return # Cannot spawn in wall or exit

        self.clones.append({
            'pos': list(pos),
            'visual_pos': [p * self.CELL_SIZE + self.CELL_SIZE / 2 for p in pos],
            'dir': (1, 0),
            'alive': True
        })
        self.step_rewards.append(1.0) # Reward for placing a clone
        # sfx: clone_spawn
        for _ in range(20):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 3)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.particles.append(Particle(
                [pos[0] * self.CELL_SIZE + self.CELL_SIZE / 2, pos[1] * self.CELL_SIZE + self.CELL_SIZE / 2],
                vel, random.randint(20, 40), self.COLOR_CLONE, self.COLOR_BG, 4, 1))

    def _place_portal(self):
        pos = tuple(self.cursor_pos)
        if self.level_data[pos] == 1: return # Cannot place in wall

        if self.portal_placement_mode == 'entry':
            self.portals['entry'] = pos
            self.portal_placement_mode = 'exit'
            # sfx: portal_entry_set
        elif self.portal_placement_mode == 'exit':
            if pos == self.portals['entry']: # Can't be same spot
                self.portals['entry'] = None
                self.portal_placement_mode = 'entry'
            else:
                self.portals['exit'] = pos
                self.portal_placement_mode = 'clear'
                # sfx: portal_exit_set
        else: # 'clear' mode
            self.portals = {'entry': None, 'exit': None}
            self.portal_placement_mode = 'entry'
            # sfx: portal_clear

    def _update_clones(self):
        for clone in self.clones:
            if not clone['alive']:
                continue

            # Interpolate visual position
            target_visual_pos = [p * self.CELL_SIZE + self.CELL_SIZE / 2 for p in clone['pos']]
            clone['visual_pos'][0] += (target_visual_pos[0] - clone['visual_pos'][0]) * 0.5
            clone['visual_pos'][1] += (target_visual_pos[1] - clone['visual_pos'][1]) * 0.5

            # Move clone logically
            next_pos = (clone['pos'][0] + clone['dir'][0], clone['pos'][1] + clone['dir'][1])

            # Portal check
            if self.portals['entry'] and self.portals['exit'] and tuple(clone['pos']) == self.portals['entry']:
                clone['pos'] = list(self.portals['exit'])
                next_pos = (clone['pos'][0] + clone['dir'][0], clone['pos'][1] + clone['dir'][1])
                # sfx: portal_travel
                for _ in range(30):
                    angle = random.uniform(0, 2 * math.pi)
                    speed = random.uniform(2, 5)
                    start_pos_px = [p * self.CELL_SIZE + self.CELL_SIZE / 2 for p in self.portals['exit']]
                    self.particles.append(Particle(start_pos_px, [math.cos(angle)*speed, math.sin(angle)*speed], 30, self.COLOR_PORTAL_OUT, self.COLOR_BG, 5))

            # Collision and interaction checks
            if not (0 <= next_pos[0] < self.GRID_W and 0 <= next_pos[1] < self.GRID_H):
                clone['alive'] = False # Out of bounds
            elif self.level_data[next_pos] == 1 or (next_pos in self.doors and not self.doors[next_pos]['open']):
                clone['dir'] = (-clone['dir'][0], -clone['dir'][1]) # Bounce off wall/closed door
                # sfx: clone_bounce
            elif next_pos in self.hazards:
                clone['alive'] = False # Hit hazard
            else:
                clone['pos'] = list(next_pos)

            # Check for events at current position
            pos_tuple = tuple(clone['pos'])
            if pos_tuple == self.exit_pos:
                self.score += 100
                self.game_over = True # Win condition
                self.level_num += 1
                # sfx: level_win
            elif pos_tuple in self.switches and not self.switches[pos_tuple]['activated']:
                self.switches[pos_tuple]['activated'] = True
                self.step_rewards.append(5.0)
                self.combo += 1
                self.combo_timer = 30 # steps
                # sfx: switch_activate
            
            if not clone['alive']:
                # sfx: clone_destroy
                pos_px = [p * self.CELL_SIZE + self.CELL_SIZE / 2 for p in clone['pos']]
                for _ in range(40):
                    angle = random.uniform(0, 2 * math.pi)
                    speed = random.uniform(1, 4)
                    self.particles.append(Particle(pos_px, [math.cos(angle)*speed, math.sin(angle)*speed], 40, self.COLOR_HAZARD, self.COLOR_BG, 5, 0))

    def _update_particles(self):
        self.particles = [p for p in self.particles if p.update()]

    def _update_switches_and_doors(self):
        for switch_pos, switch_data in self.switches.items():
            if switch_data['activated']:
                door_pos = switch_data['door_pos']
                if door_pos in self.doors and not self.doors[door_pos]['open']:
                    self.doors[door_pos]['open'] = True
                    # sfx: door_open

    def _update_combo(self):
        if self.combo_timer > 0:
            self.combo_timer -= 1
            if self.combo_timer == 0:
                if self.combo > 1:
                    reward = 10 * (self.combo - 1)
                    self.step_rewards.append(reward)
                    # sfx: combo_success
                self.combo = 0
        
    def _calculate_reward(self):
        return sum(self.step_rewards)

    def _check_termination(self):
        if self.game_over:
            return True
        
        # Check if any clones are still alive
        if any(c['alive'] for c in self.clones):
            return False
        
        # If no clones are alive, but some were created, it's a loss
        if len(self.clones) > 0:
            self.score -= 50
            self.game_over = True
            return True
            
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        self._draw_grid()
        self._draw_level_elements()
        self._draw_portals()
        self._draw_particles()
        self._draw_clones()
        self._draw_cursor()

    def _draw_grid(self):
        for x in range(0, self.WIDTH, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))

    def _draw_level_elements(self):
        # Walls
        for x in range(self.GRID_W):
            for y in range(self.GRID_H):
                if self.level_data[x, y] == 1:
                    pygame.draw.rect(self.screen, self.COLOR_WALL, (x * self.CELL_SIZE, y * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE))
        
        # Exit
        if self.exit_pos:
            pulse = (math.sin(self.steps * 0.1) + 1) / 2
            color = [int(c * (0.6 + 0.4 * pulse)) for c in self.COLOR_EXIT]
            pos_px = [p * self.CELL_SIZE for p in self.exit_pos]
            rect = pygame.Rect(pos_px[0], pos_px[1], self.CELL_SIZE, self.CELL_SIZE)
            pygame.draw.rect(self.screen, tuple(color), rect.inflate(pulse * 6, pulse * 6))

        # Switches, Doors, Hazards
        for pos, data in self.switches.items():
            color = self.COLOR_SWITCH_ON if data['activated'] else self.COLOR_SWITCH_OFF
            pygame.draw.rect(self.screen, color, (pos[0] * self.CELL_SIZE + 4, pos[1] * self.CELL_SIZE + 4, self.CELL_SIZE - 8, self.CELL_SIZE - 8), border_radius=4)
        for pos, data in self.doors.items():
            color = self.COLOR_DOOR_OPEN if data['open'] else self.COLOR_DOOR_CLOSED
            pygame.draw.rect(self.screen, color, (pos[0] * self.CELL_SIZE, pos[1] * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE))
        for pos in self.hazards:
            pygame.draw.rect(self.screen, self.COLOR_HAZARD, (pos[0] * self.CELL_SIZE + 5, pos[1] * self.CELL_SIZE + 5, self.CELL_SIZE - 10, self.CELL_SIZE - 10))

    def _draw_portals(self):
        for portal_type, pos in self.portals.items():
            if pos:
                color = self.COLOR_PORTAL_IN if portal_type == 'entry' else self.COLOR_PORTAL_OUT
                center_px = (int(pos[0] * self.CELL_SIZE + self.CELL_SIZE / 2), int(pos[1] * self.CELL_SIZE + self.CELL_SIZE / 2))
                radius = int(self.CELL_SIZE / 2 * 0.8)
                pulse = (math.sin(self.steps * 0.2 + (1 if portal_type == 'exit' else 0)) + 1) / 2
                
                for i in range(3):
                    alpha_color = (*color, int(100 * (0.2 + pulse * 0.8) * (i / 3)))
                    s = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
                    pygame.gfxdraw.filled_circle(s, radius, radius, radius - i * 2, alpha_color)
                    self.screen.blit(s, (center_px[0] - radius, center_px[1] - radius))

    def _draw_clones(self):
        for clone in self.clones:
            if clone['alive']:
                center_px = (int(clone['visual_pos'][0]), int(clone['visual_pos'][1]))
                radius = int(self.CELL_SIZE / 2 * 0.7)
                # Glow effect
                pygame.gfxdraw.filled_circle(self.screen, center_px[0], center_px[1], radius + 3, (*self.COLOR_CLONE, 50))
                pygame.gfxdraw.filled_circle(self.screen, center_px[0], center_px[1], radius, self.COLOR_CLONE)
                pygame.gfxdraw.aacircle(self.screen, center_px[0], center_px[1], radius, self.COLOR_CLONE)

    def _draw_cursor(self):
        x, y = self.cursor_pos
        rect = pygame.Rect(x * self.CELL_SIZE, y * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, rect, 2)
        
    def _draw_particles(self):
        for p in self.particles:
            p.draw(self.screen)

    def _render_ui(self):
        score_text = self.font_large.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        level_text = self.font_large.render(f"LEVEL: {self.level_num}", True, self.COLOR_TEXT)
        self.screen.blit(level_text, (self.WIDTH - level_text.get_width() - 10, 10))

        alive_clones = sum(1 for c in self.clones if c['alive'])
        clone_text = self.font_small.render(f"CLONES: {alive_clones}/{len(self.clones)}", True, self.COLOR_TEXT)
        self.screen.blit(clone_text, (10, 40))

        if self.combo > 1:
            combo_text = self.font_large.render(f"COMBO x{self.combo}", True, self.COLOR_CURSOR)
            self.screen.blit(combo_text, (self.WIDTH // 2 - combo_text.get_width() // 2, 10))
        
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "level": self.level_num,
            "clones_alive": sum(1 for c in self.clones if c['alive']),
            "combo": self.combo
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # It will not run in the evaluation environment
    os.environ["SDL_VIDEODRIVER"] = "pygame"
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Rhythm Clone Puzzler")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement = 0 # none
        space = 0
        shift = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
        if keys[pygame.K_r]:
            obs, info = env.reset()
            total_reward = 0
            
        action = [movement, space, shift]
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Pygame uses a different coordinate system for blitting numpy arrays
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Episode finished. Total reward: {total_reward}, Score: {info['score']}, Steps: {info['steps']}")
            obs, info = env.reset()
            total_reward = 0
            pygame.time.wait(2000)

        clock.tick(30) # Run at 30 FPS for smooth manual play

    env.close()