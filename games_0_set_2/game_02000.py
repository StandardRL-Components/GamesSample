
# Generated: 2025-08-27T18:55:35.537592
# Source Brief: brief_02000.md
# Brief Index: 2000

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
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
        "Controls: Arrow keys to move. Hold space to attack nearby zombies."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Escape hordes of procedurally generated zombies and reach the rescue helicopter."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    PLAYER_SIZE = 18
    PLAYER_SPEED = 4.0
    ZOMBIE_SIZE = 16
    ZOMBIE_SPEED_SLOW = 1.0
    ZOMBIE_SPEED_FAST = 1.8
    HELICOPTER_ZONE_SIZE = 60
    ATTACK_RADIUS = 50
    ATTACK_COOLDOWN = 10  # frames
    MAX_ZOMBIES = 50
    MAX_HITS = 5
    MAX_STEPS = 1800 # 60 seconds at 30fps

    # --- Colors ---
    COLOR_BG = (30, 35, 40)
    COLOR_GRID = (40, 45, 50)
    COLOR_PLAYER = (50, 255, 100)
    COLOR_PLAYER_OUTLINE = (200, 255, 220)
    COLOR_ZOMBIE_SLOW = (220, 50, 50)
    COLOR_ZOMBIE_FAST = (160, 20, 20)
    COLOR_HELICOPTER = (255, 255, 0)
    COLOR_BLOOD = (120, 0, 0)
    COLOR_ATTACK_WAVE = (255, 255, 255)
    COLOR_HIT_WAVE = (255, 50, 50)
    COLOR_UI_TEXT = (220, 220, 220)
    COLOR_ARROW = (255, 255, 0)

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
        self.font_main = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_h = pygame.font.SysFont("Arial", 50, bold=True)
        
        self.render_mode = render_mode
        self.player_pos = None
        self.helicopter_pos = None
        self.zombies = None
        self.particles = None
        self.effects = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.zombies_killed = None
        self.player_hits = None
        self.zombie_spawn_timer = None
        self.zombie_spawn_rate = None
        self.attack_cooldown_timer = None
        
        self.reset()
        # self.validate_implementation() # Optional self-check

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.player_pos = np.array([self.WIDTH / 2, self.HEIGHT / 2], dtype=np.float32)
        
        # Place helicopter away from the center
        angle = self.np_random.uniform(0, 2 * math.pi)
        dist = self.np_random.uniform(self.WIDTH * 0.4, self.WIDTH * 0.45)
        self.helicopter_pos = np.array([
            self.WIDTH / 2 + dist * math.cos(angle),
            self.HEIGHT / 2 + dist * math.sin(angle) * (self.HEIGHT/self.WIDTH)
        ], dtype=np.float32)

        self.zombies = []
        self.particles = []
        self.effects = []
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.zombies_killed = 0
        self.player_hits = 0
        self.zombie_spawn_timer = 0
        self.zombie_spawn_rate = 0.1  # Initial rate: 1 zombie every 10 frames
        self.attack_cooldown_timer = 0
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        
        # --- Handle Player Actions ---
        movement, space_held, _ = action
        self._handle_movement(movement)
        
        killed_in_step = 0
        if space_held and self.attack_cooldown_timer <= 0:
            killed_in_step = self._perform_attack()
            self.zombies_killed += killed_in_step
            reward += killed_in_step * 1.0 # +1 reward per kill
            # sfx: player_attack_swoosh

        # --- Update Game State ---
        self._update_timers()
        self._update_zombies()
        self._update_spawner()
        self._update_particles_and_effects()
        
        # --- Calculate Rewards ---
        reward += 0.01  # Small survival reward
        
        # Proximity penalty
        if self.zombies:
            min_dist = min(np.linalg.norm(self.player_pos - z['pos']) for z in self.zombies)
            if min_dist < self.ATTACK_RADIUS * 1.5:
                reward -= 0.05

        # --- Check Termination ---
        terminated = self._check_termination()
        if terminated:
            if self.player_hits >= self.MAX_HITS:
                reward -= 100 # Penalty for dying
            elif self._is_player_in_helicopter_zone():
                reward += 100 # Reward for winning
        
        self.score += reward
        self.steps += 1
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_movement(self, movement):
        direction = np.array([0.0, 0.0])
        if movement == 1: # Up
            direction[1] -= 1.0
        elif movement == 2: # Down
            direction[1] += 1.0
        elif movement == 3: # Left
            direction[0] -= 1.0
        elif movement == 4: # Right
            direction[0] += 1.0
        
        if np.linalg.norm(direction) > 0:
            direction = direction / np.linalg.norm(direction)
        
        self.player_pos += direction * self.PLAYER_SPEED
        self.player_pos[0] = np.clip(self.player_pos[0], self.PLAYER_SIZE, self.WIDTH - self.PLAYER_SIZE)
        self.player_pos[1] = np.clip(self.player_pos[1], self.PLAYER_SIZE, self.HEIGHT - self.PLAYER_SIZE)

    def _perform_attack(self):
        self.attack_cooldown_timer = self.ATTACK_COOLDOWN
        self.effects.append({'type': 'attack', 'pos': self.player_pos.copy(), 'radius': 0, 'max_radius': self.ATTACK_RADIUS, 'life': 5})
        
        killed_zombies = []
        killed_count = 0
        for zombie in self.zombies:
            if np.linalg.norm(self.player_pos - zombie['pos']) <= self.ATTACK_RADIUS:
                killed_zombies.append(zombie)
                # sfx: zombie_die
                for _ in range(15): # Blood particles
                    self.particles.append(self._create_particle(zombie['pos']))
        
        if killed_zombies:
            self.zombies = [z for z in self.zombies if z not in killed_zombies]
            killed_count = len(killed_zombies)
        
        return killed_count

    def _update_timers(self):
        if self.attack_cooldown_timer > 0:
            self.attack_cooldown_timer -= 1
        
        # Increase difficulty over time
        self.zombie_spawn_rate += 0.0003 # Equivalent to 0.01 per second at 30fps

    def _update_zombies(self):
        zombies_to_remove = []
        for zombie in self.zombies:
            direction_to_player = self.player_pos - zombie['pos']
            dist = np.linalg.norm(direction_to_player)
            
            if dist > 1:
                direction_to_player /= dist
                zombie['pos'] += direction_to_player * zombie['speed']
            
            if dist < (self.PLAYER_SIZE + self.ZOMBIE_SIZE) / 2:
                zombies_to_remove.append(zombie)
                self.player_hits += 1
                # sfx: player_hit
                self.effects.append({'type': 'hit', 'pos': self.player_pos.copy(), 'radius': 0, 'max_radius': 80, 'life': 10})

        if zombies_to_remove:
            self.zombies = [z for z in self.zombies if z not in zombies_to_remove]

    def _update_spawner(self):
        self.zombie_spawn_timer += self.zombie_spawn_rate
        if self.zombie_spawn_timer >= 1 and len(self.zombies) < self.MAX_ZOMBIES:
            self.zombie_spawn_timer -= 1
            self._spawn_zombie()

    def _spawn_zombie(self):
        # Spawn on an edge
        edge = self.np_random.integers(4)
        if edge == 0: # Top
            pos = np.array([self.np_random.uniform(0, self.WIDTH), -self.ZOMBIE_SIZE])
        elif edge == 1: # Bottom
            pos = np.array([self.np_random.uniform(0, self.WIDTH), self.HEIGHT + self.ZOMBIE_SIZE])
        elif edge == 2: # Left
            pos = np.array([-self.ZOMBIE_SIZE, self.np_random.uniform(0, self.HEIGHT)])
        else: # Right
            pos = np.array([self.WIDTH + self.ZOMBIE_SIZE, self.np_random.uniform(0, self.HEIGHT)])
        
        is_fast = self.np_random.random() < 0.2
        speed = self.ZOMBIE_SPEED_FAST if is_fast else self.ZOMBIE_SPEED_SLOW
        
        self.zombies.append({'pos': pos, 'speed': speed, 'fast': is_fast})

    def _create_particle(self, pos):
        angle = self.np_random.uniform(0, 2 * math.pi)
        speed = self.np_random.uniform(1, 4)
        velocity = np.array([math.cos(angle), math.sin(angle)]) * speed
        return {'pos': pos.copy(), 'vel': velocity, 'life': self.np_random.integers(10, 20)}

    def _update_particles_and_effects(self):
        # Update particles
        for p in self.particles:
            p['pos'] += p['vel']
            p['vel'] *= 0.95 # friction
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]
        
        # Update effects
        for e in self.effects:
            e['life'] -= 1
            if e['type'] in ['attack', 'hit']:
                e['radius'] += e['max_radius'] / (e['max_radius'] / (10 if e['type'] == 'hit' else 5))
        self.effects = [e for e in self.effects if e['life'] > 0]

    def _is_player_in_helicopter_zone(self):
        return np.linalg.norm(self.player_pos - self.helicopter_pos) < self.HELICOPTER_ZONE_SIZE / 2

    def _check_termination(self):
        if self.game_over:
            return True
        if self.player_hits >= self.MAX_HITS:
            self.game_over = True
            return True
        if self._is_player_in_helicopter_zone():
            self.game_over = True
            return True
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True
        return False

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
            "zombies_killed": self.zombies_killed,
            "player_hits": self.player_hits,
        }

    def _draw_iso_rect(self, surface, color, pos, size):
        x, y = pos
        half_size = size / 2
        points = [
            (x, y - half_size),
            (x + half_size, y),
            (x, y + half_size),
            (x - half_size, y)
        ]
        pygame.gfxdraw.aapolygon(surface, [(int(px), int(py)) for px, py in points], color)
        pygame.gfxdraw.filled_polygon(surface, [(int(px), int(py)) for px, py in points], color)

    def _render_game(self):
        # Draw grid
        for i in range(0, self.WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (i, 0), (i, self.HEIGHT))
        for i in range(0, self.HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, i), (self.WIDTH, i))

        # Draw helicopter zone
        h_pos = (int(self.helicopter_pos[0]), int(self.helicopter_pos[1]))
        pygame.gfxdraw.aacircle(self.screen, h_pos[0], h_pos[1], int(self.HELICOPTER_ZONE_SIZE/2), self.COLOR_HELICOPTER)
        pygame.gfxdraw.aacircle(self.screen, h_pos[0], h_pos[1], int(self.HELICOPTER_ZONE_SIZE/2)-2, self.COLOR_HELICOPTER)
        h_text = self.font_h.render('H', True, self.COLOR_HELICOPTER)
        self.screen.blit(h_text, h_text.get_rect(center=h_pos))
        
        # Draw particles
        for p in self.particles:
            alpha = max(0, 255 * (p['life'] / 20.0))
            color = (*self.COLOR_BLOOD, alpha)
            s = pygame.Surface((3, 3), pygame.SRCALPHA)
            s.fill(color)
            self.screen.blit(s, (int(p['pos'][0]), int(p['pos'][1])))

        # Draw zombies
        for z in self.zombies:
            color = self.COLOR_ZOMBIE_FAST if z['fast'] else self.COLOR_ZOMBIE_SLOW
            self._draw_iso_rect(self.screen, color, z['pos'], self.ZOMBIE_SIZE)

        # Draw player
        self._draw_iso_rect(self.screen, self.COLOR_PLAYER, self.player_pos, self.PLAYER_SIZE)
        self._draw_iso_rect(self.screen, self.COLOR_PLAYER_OUTLINE, self.player_pos, self.PLAYER_SIZE+2)
        self._draw_iso_rect(self.screen, self.COLOR_PLAYER, self.player_pos, self.PLAYER_SIZE) # Redraw to be on top

        # Draw effects
        for e in self.effects:
            alpha = max(0, 255 * (e['life'] / 10.0))
            color = self.COLOR_ATTACK_WAVE if e['type'] == 'attack' else self.COLOR_HIT_WAVE
            if e['radius'] > 1:
                pygame.gfxdraw.aacircle(self.screen, int(e['pos'][0]), int(e['pos'][1]), int(e['radius']), (*color, alpha))
                pygame.gfxdraw.aacircle(self.screen, int(e['pos'][0]), int(e['pos'][1]), int(e['radius'])-1, (*color, alpha))

    def _render_ui(self):
        # Kills
        kills_text = self.font_main.render(f'KILLS: {self.zombies_killed}', True, self.COLOR_UI_TEXT)
        self.screen.blit(kills_text, (10, 5))
        
        # Hits
        hits_text = self.font_main.render(f'HITS: {self.player_hits}/{self.MAX_HITS}', True, self.COLOR_UI_TEXT)
        self.screen.blit(hits_text, (10, 25))

        # Timer
        time_left = (self.MAX_STEPS - self.steps) // 30
        time_text = self.font_main.render(f'TIME: {time_left}', True, self.COLOR_UI_TEXT)
        self.screen.blit(time_text, (self.WIDTH - time_text.get_width() - 10, 5))
        
        # Helicopter arrow
        direction_to_heli = self.helicopter_pos - self.player_pos
        angle = math.atan2(direction_to_heli[1], direction_to_heli[0])
        arrow_center = (self.WIDTH / 2, 20)
        arrow_points = [
            (arrow_center[0] + 15 * math.cos(angle), arrow_center[1] + 15 * math.sin(angle)),
            (arrow_center[0] + 5 * math.cos(angle + math.pi/2), arrow_center[1] + 5 * math.sin(angle + math.pi/2)),
            (arrow_center[0] + 5 * math.cos(angle - math.pi/2), arrow_center[1] + 5 * math.sin(angle - math.pi/2)),
        ]
        pygame.gfxdraw.aapolygon(self.screen, [(int(p[0]), int(p[1])) for p in arrow_points], self.COLOR_ARROW)
        pygame.gfxdraw.filled_polygon(self.screen, [(int(p[0]), int(p[1])) for p in arrow_points], self.COLOR_ARROW)

    def close(self):
        pygame.font.quit()
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    import os
    # Set the video driver to a dummy one if not on a display
    if os.environ.get('SDL_VIDEODRIVER', '') == 'dummy':
        pass
    else:
        # If on a display, set up a window to see the game
        pygame.display.set_caption("Zombie Survival")
        screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    done = False
    
    # Game loop
    while not done:
        # Human controls
        keys = pygame.key.get_pressed()
        mov = 0 # No-op
        if keys[pygame.K_UP]: mov = 1
        elif keys[pygame.K_DOWN]: mov = 2
        elif keys[pygame.K_LEFT]: mov = 3
        elif keys[pygame.K_RIGHT]: mov = 4
        
        space = 1 if keys[pygame.K_SPACE] else 0
        shift = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [mov, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Render to the screen if not in dummy mode
        if os.environ.get('SDL_VIDEODRIVER', '') != 'dummy':
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()

        # Event handling (for closing the window)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        env.clock.tick(30) # Limit to 30 FPS

    print(f"Game Over! Final Score: {info['score']:.2f}, Kills: {info['zombies_killed']}")
    env.close()