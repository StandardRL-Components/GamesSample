
# Generated: 2025-08-28T01:59:15.734993
# Source Brief: brief_04302.md
# Brief Index: 4302

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
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
        "Controls: Use ↑↓←→ to apply a global push to all crystals in the cavern."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "An isometric puzzle game. Push glowing crystals into their matching slots before time runs out."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.GRID_WIDTH, self.GRID_HEIGHT = 22, 16
        self.NUM_CRYSTALS = 10
        self.TIME_LIMIT_SECONDS = 60
        self.MAX_STEPS = self.TIME_LIMIT_SECONDS * self.FPS

        # Visual constants
        self.TILE_W, self.TILE_H = 24, 12
        self.CRYSTAL_H = 20
        self.ORIGIN_X, self.ORIGIN_Y = self.WIDTH // 2, 80

        # Colors
        self.COLOR_BG = (15, 20, 35)
        self.COLOR_GRID = (25, 35, 55)
        self.COLOR_WALL = (40, 50, 70)
        self.CRYSTAL_PALETTE = [
            (255, 80, 80), (80, 255, 80), (80, 80, 255), (255, 255, 80),
            (80, 255, 255), (255, 80, 255), (255, 160, 80), (80, 255, 160),
            (160, 80, 255), (220, 220, 220)
        ]

        # EXACT spaces:
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 28)
        self.font_msg = pygame.font.Font(None, 50)
        
        # Initialize state variables
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_remaining = 0
        self.crystals = []
        self.slots = []
        self.grid = np.zeros((self.GRID_WIDTH, self.GRID_HEIGHT), dtype=int)
        self.particles = []
        self.slotted_count = 0
        self.prev_total_dist = 0
        self.np_random = None

        self.reset()
        # self.validate_implementation() # Optional validation call

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_remaining = self.MAX_STEPS
        self.slotted_count = 0
        self.particles.clear()
        
        self._generate_level()
        self.prev_total_dist = self._calculate_total_distance()

        return self._get_observation(), self._get_info()

    def _generate_level(self):
        self.grid.fill(0)
        self.grid[0, :] = 1
        self.grid[-1, :] = 1
        self.grid[:, 0] = 1
        self.grid[:, -1] = 1
        
        self.slots.clear()
        self.crystals.clear()
        
        occupied_coords = set()
        
        colors = self.CRYSTAL_PALETTE[:]
        self.np_random.shuffle(colors)

        for i in range(self.NUM_CRYSTALS):
            while True:
                pos = (
                    self.np_random.integers(2, self.GRID_WIDTH - 2),
                    self.np_random.integers(2, self.GRID_HEIGHT - 2)
                )
                if pos not in occupied_coords:
                    occupied_coords.add(pos)
                    self.slots.append({'pos': pos, 'color': colors[i], 'id': i})
                    break

        for i in range(self.NUM_CRYSTALS):
            while True:
                pos = (
                    self.np_random.integers(1, self.GRID_WIDTH - 1),
                    self.np_random.integers(1, self.GRID_HEIGHT - 1)
                )
                if pos not in occupied_coords:
                    # Ensure it has at least one empty neighbor to move into
                    neighbors = [(pos[0]+dx, pos[1]+dy) for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]]
                    can_move = False
                    for nx, ny in neighbors:
                        if 1 <= nx < self.GRID_WIDTH -1 and 1 <= ny < self.GRID_HEIGHT - 1 and (nx, ny) not in occupied_coords:
                            can_move = True
                            break
                    if can_move:
                        occupied_coords.add(pos)
                        screen_pos = self._iso_to_screen(pos[0], pos[1])
                        self.crystals.append({
                            'pos': pos, 'color': colors[i], 'id': i, 'slotted': False,
                            'render_pos': list(screen_pos), 'target_pos': list(screen_pos)
                        })
                        self.grid[pos] = i + 2 # Mark grid with crystal ID
                        break

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0.0
        
        if not self.game_over:
            self.steps += 1
            self.time_remaining -= 1

            if movement != 0:
                # sfx: push_attempt
                self._handle_push(movement)

            self._update_animations()
            
            newly_slotted = self._check_and_slot_crystals()
            if newly_slotted > 0:
                reward += newly_slotted * 10.0
                self.slotted_count += newly_slotted
                # sfx: slot_success

            current_total_dist = self._calculate_total_distance()
            dist_change = self.prev_total_dist - current_total_dist
            reward += dist_change * 0.1
            self.prev_total_dist = current_total_dist
        
        terminated = self._check_termination()
        if terminated and not self.game_over:
            self.game_over = True
            if self.slotted_count == self.NUM_CRYSTALS:
                reward += 100.0 # Win
                # sfx: win_jingle
            else:
                reward -= 100.0 # Lose
                # sfx: lose_fanfare

        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_push(self, movement):
        # 1=up, 2=down, 3=left, 4=right
        direction_map = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
        dx, dy = direction_map[movement]

        # Sort crystals to push from the 'back' of the direction of movement
        # This allows for correct non-chaining pushes in a single pass
        sort_key = lambda c: -(c['pos'][0] * dx + c['pos'][1] * dy)
        sorted_crystals = sorted([c for c in self.crystals if not c['slotted']], key=sort_key)
        
        moved = False
        for crystal in sorted_crystals:
            if crystal['slotted']: continue

            old_pos = crystal['pos']
            new_pos = (old_pos[0] + dx, old_pos[1] + dy)

            if self.grid[new_pos] == 0: # If target is empty
                self.grid[old_pos] = 0
                self.grid[new_pos] = crystal['id'] + 2
                crystal['pos'] = new_pos
                crystal['target_pos'] = self._iso_to_screen(new_pos[0], new_pos[1])
                moved = True
        
        if moved:
            # sfx: push_succeed
            pass

    def _update_animations(self):
        # Update crystal positions (lerp for smooth movement)
        for c in self.crystals:
            c['render_pos'][0] += (c['target_pos'][0] - c['render_pos'][0]) * 0.4
            c['render_pos'][1] += (c['target_pos'][1] - c['render_pos'][1]) * 0.4

        # Update particles
        new_particles = []
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            if p['life'] > 0:
                new_particles.append(p)
        self.particles = new_particles

    def _check_and_slot_crystals(self):
        newly_slotted = 0
        for crystal in self.crystals:
            if crystal['slotted']: continue
            for slot in self.slots:
                if crystal['id'] == slot['id'] and crystal['pos'] == slot['pos']:
                    crystal['slotted'] = True
                    newly_slotted += 1
                    # Create particle burst
                    sx, sy = self._iso_to_screen(crystal['pos'][0], crystal['pos'][1])
                    for _ in range(30):
                        angle = self.np_random.uniform(0, 2 * math.pi)
                        speed = self.np_random.uniform(1, 4)
                        self.particles.append({
                            'pos': [sx, sy - self.CRYSTAL_H / 2],
                            'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                            'life': self.np_random.integers(15, 30),
                            'color': crystal['color'],
                            'size': self.np_random.uniform(2, 4)
                        })
                    break
        return newly_slotted

    def _calculate_total_distance(self):
        total_dist = 0
        for crystal in self.crystals:
            if crystal['slotted']: continue
            slot_pos = self.slots[crystal['id']]['pos']
            dist = abs(crystal['pos'][0] - slot_pos[0]) + abs(crystal['pos'][1] - slot_pos[1])
            total_dist += dist
        return total_dist

    def _check_termination(self):
        return self.time_remaining <= 0 or self.slotted_count == self.NUM_CRYSTALS

    def _iso_to_screen(self, x, y):
        sx = self.ORIGIN_X + (x - y) * self.TILE_W
        sy = self.ORIGIN_Y + (x + y) * self.TILE_H
        return sx, sy

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render grid floor, slots, and crystals in correct Z-order
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                # Render floor tile
                sx, sy = self._iso_to_screen(x, y)
                points = [
                    (sx, sy), (sx + self.TILE_W, sy + self.TILE_H),
                    (sx, sy + 2 * self.TILE_H), (sx - self.TILE_W, sy + self.TILE_H)
                ]
                color = self.COLOR_WALL if self.grid[x, y] == 1 else self.COLOR_GRID
                pygame.gfxdraw.filled_polygon(self.screen, points, color)
                
                # Render slots
                for slot in self.slots:
                    if slot['pos'] == (x, y):
                        slot_color = tuple(c * 0.6 for c in slot['color'])
                        pygame.gfxdraw.filled_polygon(self.screen, points, slot_color)
                        pygame.gfxdraw.aapolygon(self.screen, points, slot['color'])
                        break
        
        # Render crystals
        sorted_crystals = sorted(self.crystals, key=lambda c: c['pos'][0] + c['pos'][1])
        for crystal in sorted_crystals:
            self._render_crystal(crystal)

        # Render particles on top
        self._render_particles()

    def _render_crystal(self, crystal):
        sx, sy = crystal['render_pos']
        h = self.CRYSTAL_H
        w = self.TILE_W
        
        color = crystal['color']
        top_color = color
        left_color = tuple(c * 0.6 for c in color)
        right_color = tuple(c * 0.8 for c in color)

        # Points for the cube
        p_top_mid = (sx, sy - h)
        p_top_r = (sx + w, sy - h + self.TILE_H)
        p_top_l = (sx - w, sy - h + self.TILE_H)
        p_top_b = (sx, sy - h + 2 * self.TILE_H)
        
        p_bot_mid = (sx, sy)
        p_bot_r = (sx + w, sy + self.TILE_H)
        p_bot_l = (sx - w, sy + self.TILE_H)

        # Draw faces
        # Top face
        top_points = [p_top_l, p_top_mid, p_top_r, p_top_b]
        pygame.gfxdraw.filled_polygon(self.screen, top_points, top_color)
        
        # Left face
        left_points = [p_bot_l, p_bot_mid, p_top_mid, p_top_l]
        pygame.gfxdraw.filled_polygon(self.screen, left_points, left_color)
        
        # Right face
        right_points = [p_bot_r, p_bot_mid, p_top_mid, p_top_r]
        pygame.gfxdraw.filled_polygon(self.screen, right_points, right_color)

        # Outline
        if crystal['slotted']:
            # Add a glow/lock effect
            pygame.gfxdraw.aapolygon(self.screen, top_points, (255, 255, 255))
        else:
            pygame.gfxdraw.aapolygon(self.screen, top_points, (0,0,0))
            pygame.gfxdraw.aapolygon(self.screen, left_points, (0,0,0))
            pygame.gfxdraw.aapolygon(self.screen, right_points, (0,0,0))

    def _render_particles(self):
        for p in self.particles:
            pygame.draw.circle(self.screen, p['color'], p['pos'], p['size'] * (p['life'] / 30.0))

    def _render_ui(self):
        # Timer
        time_sec = self.time_remaining / self.FPS
        time_str = f"{int(time_sec // 60):02}:{int(time_sec % 60):02}"
        time_perc = self.time_remaining / self.MAX_STEPS
        if time_perc < 0.2: color = (255, 50, 50)
        elif time_perc < 0.5: color = (255, 255, 50)
        else: color = (50, 255, 50)
        
        text_surf = self.font_ui.render(time_str, True, color)
        text_rect = text_surf.get_rect(topright=(self.WIDTH - 10, 10))
        pygame.draw.rect(self.screen, (0,0,0,128), text_rect.inflate(10, 6))
        self.screen.blit(text_surf, text_rect)

        # Slotted count
        slot_str = f"SLOTS: {self.slotted_count} / {self.NUM_CRYSTALS}"
        slot_surf = self.font_ui.render(slot_str, True, (200, 200, 255))
        slot_rect = slot_surf.get_rect(bottomleft=(10, self.HEIGHT - 10))
        pygame.draw.rect(self.screen, (0,0,0,128), slot_rect.inflate(10, 6))
        self.screen.blit(slot_surf, slot_rect)

        # Game Over Message
        if self.game_over:
            if self.slotted_count == self.NUM_CRYSTALS:
                msg, msg_color = "COMPLETE", (100, 255, 100)
            else:
                msg, msg_color = "TIME UP", (255, 100, 100)
            
            msg_surf = self.font_msg.render(msg, True, msg_color)
            msg_rect = msg_surf.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            
            # Draw a semi-transparent background for the text
            bg_rect = msg_rect.inflate(40, 20)
            s = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
            s.fill((0, 0, 0, 180))
            self.screen.blit(s, bg_rect)
            
            self.screen.blit(msg_surf, msg_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining": self.time_remaining,
            "slotted_count": self.slotted_count,
        }

    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        print("Running implementation validation...")
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    
    # To run validation:
    # env.validate_implementation()
    
    obs, info = env.reset()
    done = False
    
    # Pygame setup for human play
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Crystal Cavern")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    action = env.action_space.sample()
    action.fill(0) # Start with no-op

    print("\n" + "="*30)
    print("CRYSTAL CAVERN")
    print("="*30)
    print(env.game_description)
    print(env.user_guide)
    print("="*30 + "\n")

    while running:
        # Human input mapping
        movement = 0 # No-op
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space = 1 if keys[pygame.K_SPACE] else 0
        shift = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = np.array([movement, space, shift])

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    print("Resetting environment.")
                    obs, info = env.reset()
                    total_reward = 0
                if event.key == pygame.K_ESCAPE:
                    running = False

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}, Info: {info}")
            pygame.time.wait(2000) # Pause for 2 seconds on game over
            obs, info = env.reset()
            total_reward = 0

        clock.tick(env.FPS)

    env.close()