import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T13:15:43.272086
# Source Brief: brief_01628.md
# Brief Index: 1628
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    game_description = (
        "Absorb colored energy streams and route them to matching portals to stabilize the system. "
        "Manage your energy carefully to avoid a total collapse."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move. "
        "Press spacebar near a portal to route a nearby energy stream to it."
    )
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    PLAYER_SIZE = 12
    PLAYER_SPEED = 8
    PORTAL_RADIUS = 20
    NUM_PORTALS = 4
    MAX_ENERGY = 200.0
    INITIAL_ENERGY = 100.0
    VICTORY_STABILIZE_DURATION = 500
    MAX_EPISODE_STEPS = 5000

    # Colors
    COLOR_BG = (10, 20, 30)
    COLOR_GRID = (25, 35, 45)
    COLOR_PLAYER = (255, 255, 255)
    COLOR_PLAYER_GLOW = (200, 200, 255)

    COLORS_STREAM = {
        "red": (255, 50, 50),
        "green": (50, 255, 50),
        "blue": (50, 50, 255),
    }
    COLORS_PORTAL = {
        "red": (200, 0, 0),
        "green": (0, 200, 0),
        "blue": (0, 0, 200),
    }
    COLOR_PORTAL_INACTIVE = (80, 80, 80)
    COLOR_UI_TEXT = (220, 220, 220)
    COLOR_ENERGY_BAR = (40, 200, 220)
    COLOR_ENERGY_BAR_BG = (50, 50, 50)
    COLOR_SUCCESS = (180, 255, 180)
    COLOR_FAILURE = (255, 180, 180)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 16)
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)

        self.player_pos = np.zeros(2, dtype=np.float32)
        self.portals = []
        self.energy_streams = []
        self.vfx = []
        
        self.steps = 0
        self.score = 0
        self.energy = 0.0
        self.energy_drain_rate = 0.0
        self.victory_counter = 0
        self.last_space_state = 0
        
        # self.reset() is called by the environment wrapper, no need to call it here.

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.player_pos = np.array([self.WIDTH / 2, self.HEIGHT / 2], dtype=np.float32)
        
        self.steps = 0
        self.score = 0
        self.energy = self.INITIAL_ENERGY
        self.energy_drain_rate = 0.01
        self.victory_counter = 0
        self.last_space_state = 0

        self._initialize_portals()
        self.energy_streams = []
        self.vfx = []

        # Spawn initial streams
        for _ in range(5):
            self._spawn_stream()

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        reward = 0
        
        self._handle_movement(movement)

        space_pressed = space_held and not self.last_space_state
        if space_pressed:
            activation_reward = self._handle_portal_activation()
            reward += activation_reward
        self.last_space_state = space_held

        self._update_game_state()

        # Continuous reward based on energy level
        if self.energy > self.MAX_ENERGY / 2:
            reward += 0.01
        else:
            reward -= 0.01

        terminated = self._check_termination()
        
        if terminated:
            if self.energy <= 0:
                reward -= 100 # Failure penalty
            elif self.victory_counter >= self.VICTORY_STABILIZE_DURATION:
                reward += 100 # Victory bonus
        
        truncated = self.steps >= self.MAX_EPISODE_STEPS

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info(),
        )

    def _initialize_portals(self):
        self.portals = []
        min_dist = self.PORTAL_RADIUS * 4
        while len(self.portals) < self.NUM_PORTALS:
            pos = np.array([
                self.np_random.uniform(self.PORTAL_RADIUS * 2, self.WIDTH - self.PORTAL_RADIUS * 2),
                self.np_random.uniform(self.PORTAL_RADIUS * 2, self.HEIGHT - self.PORTAL_RADIUS * 2)
            ])
            
            too_close = False
            for p in self.portals:
                if np.linalg.norm(pos - p['pos']) < min_dist:
                    too_close = True
                    break
            if not too_close:
                color_name = self.np_random.choice(list(self.COLORS_STREAM.keys()))
                self.portals.append({
                    'pos': pos,
                    'color_name': color_name,
                    'color_val': self.COLORS_PORTAL[color_name],
                    'radius': self.PORTAL_RADIUS
                })

    def _handle_movement(self, movement):
        direction = np.zeros(2, dtype=np.float32)
        if movement == 1: direction[1] = -1 # Up
        elif movement == 2: direction[1] = 1 # Down
        elif movement == 3: direction[0] = -1 # Left
        elif movement == 4: direction[0] = 1 # Right
        
        if np.linalg.norm(direction) > 0:
            self.player_pos += direction * self.PLAYER_SPEED
        
        self.player_pos[0] = np.clip(self.player_pos[0], self.PLAYER_SIZE, self.WIDTH - self.PLAYER_SIZE)
        self.player_pos[1] = np.clip(self.player_pos[1], self.PLAYER_SIZE, self.HEIGHT - self.PLAYER_SIZE)

    def _handle_portal_activation(self):
        # Find closest portal
        closest_portal = None
        min_dist = float('inf')
        for portal in self.portals:
            dist = np.linalg.norm(self.player_pos - portal['pos'])
            if dist < min_dist:
                min_dist = dist
                closest_portal = portal
        
        # Check if player is close enough to activate
        if closest_portal and min_dist < closest_portal['radius'] + self.PLAYER_SIZE:
            # Find closest energy stream particle
            closest_particle = None
            min_particle_dist = float('inf')
            for stream in self.energy_streams:
                for particle in stream['particles']:
                    dist = np.linalg.norm(self.player_pos - particle['pos'])
                    if dist < min_particle_dist:
                        min_particle_dist = dist
                        closest_particle = particle

            if closest_particle:
                # Match colors
                if closest_particle['color_name'] == closest_portal['color_name']:
                    # Correct match
                    self.energy = min(self.MAX_ENERGY, self.energy + 25)
                    self.score += 1
                    self.vfx.append(self._create_shockwave(closest_portal['pos'], self.COLOR_SUCCESS, 30))
                    # sfx: positive chime
                    return 10.0
                else:
                    # Incorrect match
                    self.energy -= 20
                    self.vfx.append(self._create_shockwave(closest_portal['pos'], self.COLOR_FAILURE, 30))
                    # sfx: negative buzz
                    return -10.0
        return 0.0

    def _update_game_state(self):
        self.steps += 1
        
        # Passive energy drain
        self.energy -= self.energy_drain_rate
        self.energy = max(0, self.energy)

        # Increase difficulty
        if self.steps > 0 and self.steps % 500 == 0:
            self.energy_drain_rate += 0.001

        # Victory condition counter
        if self.energy >= self.MAX_ENERGY:
            self.victory_counter += 1
        else:
            self.victory_counter = 0
            
        # Update and spawn streams
        self._update_streams()
        if self.steps % 75 == 0:
            self._spawn_stream()
            
        # Update VFX
        self.vfx = [v for v in self.vfx if v['life'] > 0]
        for v in self.vfx:
            v['life'] -= 1
            if v['type'] == 'shockwave':
                v['radius'] += v['speed']

    def _check_termination(self):
        return (
            self.energy <= 0 or 
            self.victory_counter >= self.VICTORY_STABILIZE_DURATION
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
            "energy": self.energy,
            "victory_progress": self.victory_counter / self.VICTORY_STABILIZE_DURATION
        }

    def _render_game(self):
        # Draw grid
        for x in range(0, self.WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))

        # Draw streams
        for stream in self.energy_streams:
            for i, p in enumerate(stream['particles']):
                alpha = int(255 * (i / len(stream['particles'])))
                color = (*p['color_val'], alpha)
                temp_surf = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, color, (p['size'], p['size']), p['size'])
                self.screen.blit(temp_surf, (int(p['pos'][0] - p['size']), int(p['pos'][1] - p['size'])))

        # Draw portals
        for portal in self.portals:
            p_pos = (int(portal['pos'][0]), int(portal['pos'][1]))
            pygame.gfxdraw.filled_circle(self.screen, p_pos[0], p_pos[1], portal['radius'], self.COLOR_PORTAL_INACTIVE)
            pygame.gfxdraw.aacircle(self.screen, p_pos[0], p_pos[1], portal['radius'], portal['color_val'])
            pygame.gfxdraw.aacircle(self.screen, p_pos[0], p_pos[1], portal['radius'] - 1, portal['color_val'])

        # Draw VFX
        for v in self.vfx:
            if v['type'] == 'shockwave':
                alpha = int(255 * (v['life']/v['max_life']))
                if alpha > 0:
                    pygame.gfxdraw.aacircle(self.screen, int(v['pos'][0]), int(v['pos'][1]), int(v['radius']), (*v['color'], alpha))
                    pygame.gfxdraw.aacircle(self.screen, int(v['pos'][0]), int(v['pos'][1]), int(v['radius']-1), (*v['color'], alpha))

        # Draw player
        px, py = int(self.player_pos[0]), int(self.player_pos[1])
        # Glow effect
        for i in range(4, 0, -1):
            alpha = 80 - i * 15
            glow_radius = self.PLAYER_SIZE + i * 2
            pygame.gfxdraw.filled_circle(self.screen, px, py, glow_radius, (*self.COLOR_PLAYER_GLOW, alpha))
        # Core
        player_rect = pygame.Rect(px - self.PLAYER_SIZE/2, py - self.PLAYER_SIZE/2, self.PLAYER_SIZE, self.PLAYER_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=2)

    def _render_ui(self):
        # Energy Bar
        bar_width = 300
        bar_height = 20
        bar_x = (self.WIDTH - bar_width) / 2
        bar_y = 10
        energy_ratio = self.energy / self.MAX_ENERGY
        
        pygame.draw.rect(self.screen, self.COLOR_ENERGY_BAR_BG, (bar_x, bar_y, bar_width, bar_height), border_radius=5)
        pygame.draw.rect(self.screen, self.COLOR_ENERGY_BAR, (bar_x, bar_y, bar_width * energy_ratio, bar_height), border_radius=5)
        
        energy_text = self.font_small.render(f"ENERGY", True, self.COLOR_UI_TEXT)
        self.screen.blit(energy_text, (bar_x - energy_text.get_width() - 10, bar_y + 2))

        # Score
        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (15, 10))

        # Steps
        steps_text = self.font_small.render(f"STEPS: {self.steps}/{self.MAX_EPISODE_STEPS}", True, self.COLOR_UI_TEXT)
        self.screen.blit(steps_text, (self.WIDTH - steps_text.get_width() - 15, 15))
        
        # Victory progress
        if self.victory_counter > 0:
            stabilize_text = self.font_small.render(f"STABILIZING: {self.victory_counter}/{self.VICTORY_STABILIZE_DURATION}", True, self.COLOR_SUCCESS)
            self.screen.blit(stabilize_text, (bar_x + bar_width + 10, bar_y + 2))

    def _spawn_stream(self):
        color_name = self.np_random.choice(list(self.COLORS_STREAM.keys()))
        side = self.np_random.choice(['left', 'right', 'top', 'bottom'])
        
        if side == 'left':
            start_pos = np.array([0.0, self.np_random.uniform(0, self.HEIGHT)])
            velocity = np.array([self.np_random.uniform(1.5, 3.0), self.np_random.uniform(-0.5, 0.5)])
        elif side == 'right':
            start_pos = np.array([float(self.WIDTH), self.np_random.uniform(0, self.HEIGHT)])
            velocity = np.array([self.np_random.uniform(-3.0, -1.5), self.np_random.uniform(-0.5, 0.5)])
        elif side == 'top':
            start_pos = np.array([self.np_random.uniform(0, self.WIDTH), 0.0])
            velocity = np.array([self.np_random.uniform(-0.5, 0.5), self.np_random.uniform(1.5, 3.0)])
        else: # bottom
            start_pos = np.array([self.np_random.uniform(0, self.WIDTH), float(self.HEIGHT)])
            velocity = np.array([self.np_random.uniform(-0.5, 0.5), self.np_random.uniform(-3.0, -1.5)])
        
        self.energy_streams.append({
            'particles': [],
            'start_pos': start_pos,
            'velocity': velocity,
            'color_name': color_name,
            'color_val': self.COLORS_STREAM[color_name],
            'spawn_timer': 0,
            'max_particles': 30
        })

    def _update_streams(self):
        for stream in self.energy_streams:
            stream['spawn_timer'] += 1
            if stream['spawn_timer'] >= 2 and len(stream['particles']) < stream['max_particles']:
                stream['spawn_timer'] = 0
                new_particle_pos = stream['particles'][0]['pos'].copy() if stream['particles'] else stream['start_pos'].copy()
                stream['particles'].insert(0, {
                    'pos': new_particle_pos,
                    'color_name': stream['color_name'],
                    'color_val': stream['color_val'],
                    'size': self.np_random.integers(4, 7),
                })

            for i, particle in enumerate(stream['particles']):
                # Move particle
                particle['pos'] += stream['velocity']
                # Particles further down the tail move slower
                particle['pos'] += stream['velocity'] * 0.02 * i
            
            # Remove off-screen particles from the end of the tail
            if stream['particles']:
                p = stream['particles'][-1]
                if not (0 <= p['pos'][0] <= self.WIDTH and 0 <= p['pos'][1] <= self.HEIGHT):
                    stream['particles'].pop()
        
        # Remove empty streams
        self.energy_streams = [s for s in self.energy_streams if s['particles']]
    
    def _create_shockwave(self, pos, color, max_life):
        return {
            'type': 'shockwave',
            'pos': pos,
            'color': color,
            'radius': self.PORTAL_RADIUS,
            'speed': 1.5,
            'life': max_life,
            'max_life': max_life,
        }
        
    def close(self):
        pygame.font.quit()
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # It will not run in a headless environment
    if os.environ.get("SDL_VIDEODRIVER") == "dummy":
        print("Cannot run interactive test in a headless environment. Skipping.")
    else:
        env = GameEnv()
        obs, info = env.reset()
        done = False
        
        # Pygame setup for manual play
        render_screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
        pygame.display.set_caption("Energy Landscape")
        clock = pygame.time.Clock()

        total_reward = 0
        
        while not done:
            movement = 0 # No-op
            space = 0
            shift = 0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True

            keys = pygame.key.get_pressed()
            if keys[pygame.K_w] or keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_s] or keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_a] or keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_d] or keys[pygame.K_RIGHT]: movement = 4
            
            if keys[pygame.K_SPACE]: space = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
            
            action = [movement, space, shift]
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward

            # Render the observation to the display
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            render_screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            clock.tick(30) # Run at 30 FPS

        print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}, Steps: {info['steps']}")
        env.close()