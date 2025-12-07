
# Generated: 2025-08-27T20:34:36.391797
# Source Brief: brief_02508.md
# Brief Index: 2508

        
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
    """
    A top-down arcade shooter where the player must survive against hordes of zombies for 60 seconds.
    The player controls a crosshair to aim and fires a piercing laser from a central turret.
    Zombies continuously spawn from the edges of the screen and move towards the turret.
    The game's difficulty increases over time with more and faster zombies.
    """
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrow keys to move the crosshair. Press space to fire your laser."
    )

    game_description = (
        "Zap hordes of procedurally generated zombies in a top-down arcade shooter before they overwhelm you."
    )

    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_TIME = 60  # seconds
        self.MAX_STEPS = self.MAX_TIME * self.FPS
        
        self.PLAYER_POS = (self.WIDTH // 2, self.HEIGHT // 2)
        self.PLAYER_RADIUS = 15
        self.ZOMBIE_RADIUS = 8
        self.MAX_ZOMBIES = 50
        self.CROSSHAIR_SPEED = 12

        # Colors
        self.COLOR_BG = (20, 20, 30)
        self.COLOR_PLAYER = (0, 255, 128)
        self.COLOR_ZOMBIE = (255, 50, 50)
        self.COLOR_PROJECTILE = (255, 255, 0)
        self.COLOR_CROSSHAIR = (0, 255, 255)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_PARTICLE = (255, 100, 100)

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", 24, bold=True)
        
        # State variables (initialized in reset)
        self.steps = None
        self.score = None
        self.time_remaining = None
        self.game_over = None
        self.crosshair_pos = None
        self.zombies = None
        self.zombie_speed = None
        self.zombie_base_count = None
        self.particles = None
        self.laser_visuals = None
        self.last_space_state = None
        self.rng = None
        
        self.reset()
        # self.validate_implementation() # Uncomment for self-check

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.rng = np.random.default_rng(seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_remaining = self.MAX_STEPS
        
        self.crosshair_pos = np.array([self.WIDTH / 2, self.HEIGHT / 2], dtype=float)
        self.zombies = []
        self.zombie_speed = 1.0
        self.zombie_base_count = 10
        
        self.particles = []
        self.laser_visuals = []
        self.last_space_state = False

        while len(self.zombies) < self.zombie_base_count:
            self._spawn_zombie()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
        
        self.clock.tick(self.FPS)
        reward = 0.0
        terminated = False

        # 1. Handle player input
        self._handle_input(action)

        # 2. Fire weapon (if triggered on key press)
        space_held = action[1] == 1
        if space_held and not self.last_space_state:
            # Sound: Laser fire
            fire_reward = self._fire_weapon()
            reward += fire_reward
        self.last_space_state = space_held

        # 3. Update game state
        self._update_zombies()
        self._update_particles()
        self._update_laser_visuals()
        self._manage_difficulty()
        self._respawn_zombies()
        
        # 4. Check for game over conditions
        if self._check_player_collision():
            # Sound: Player death/game over
            self.game_over = True
            terminated = True
            reward = -100.0
        else:
            reward += 0.1 # Survival reward

        # 5. Update timers and check for win condition
        self.steps += 1
        self.time_remaining -= 1
        if self.time_remaining <= 0:
            # Sound: Victory fanfare
            self.game_over = True
            terminated = True
            reward = 100.0
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, action):
        movement = action[0]
        if movement == 1: self.crosshair_pos[1] -= self.CROSSHAIR_SPEED # Up
        elif movement == 2: self.crosshair_pos[1] += self.CROSSHAIR_SPEED # Down
        elif movement == 3: self.crosshair_pos[0] -= self.CROSSHAIR_SPEED # Left
        elif movement == 4: self.crosshair_pos[0] += self.CROSSHAIR_SPEED # Right
        
        self.crosshair_pos[0] = np.clip(self.crosshair_pos[0], 0, self.WIDTH)
        self.crosshair_pos[1] = np.clip(self.crosshair_pos[1], 0, self.HEIGHT)

    def _fire_weapon(self):
        fire_reward = 0
        killed_zombies = []
        
        self.laser_visuals.append({
            'start': self.PLAYER_POS, 'end': tuple(self.crosshair_pos), 'life': 3
        })

        p1 = np.array(self.PLAYER_POS)
        p2 = self.crosshair_pos
        
        if np.array_equal(p1, p2): return 0

        line_vec = p2 - p1
        line_len_sq = np.dot(line_vec, line_vec)

        for i, zombie in enumerate(self.zombies):
            p3 = zombie['pos']
            # Point-to-line distance
            dist = np.linalg.norm(np.cross(line_vec, p1 - p3)) / np.linalg.norm(line_vec)

            if dist < self.ZOMBIE_RADIUS:
                # Check if zombie is on the segment between player and crosshair
                dot_product = np.dot(p3 - p1, line_vec)
                if 0 <= dot_product <= line_len_sq:
                    killed_zombies.append(i)
                    fire_reward += 1
                    self.score += 1
                    self._create_explosion(zombie['pos'])
                    # Sound: Zombie death/squish

        for i in sorted(killed_zombies, reverse=True):
            del self.zombies[i]
        
        return fire_reward

    def _create_explosion(self, pos):
        num_particles = self.rng.integers(15, 25)
        for _ in range(num_particles):
            angle = self.rng.uniform(0, 2 * math.pi)
            speed = self.rng.uniform(1, 4)
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': self.rng.integers(10, 20),
                'max_life': 20
            })

    def _update_zombies(self):
        for zombie in self.zombies:
            direction = np.array(self.PLAYER_POS) - zombie['pos']
            dist = np.linalg.norm(direction)
            if dist > 0:
                direction /= dist
            zombie['pos'] += direction * self.zombie_speed
            zombie['anim_offset'] = (zombie['anim_offset'] + 0.2) % (2 * math.pi)

    def _check_player_collision(self):
        for zombie in self.zombies:
            dist = np.linalg.norm(np.array(self.PLAYER_POS) - zombie['pos'])
            if dist < self.PLAYER_RADIUS + self.ZOMBIE_RADIUS:
                return True
        return False

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][0] *= 0.95
            p['vel'][1] *= 0.95
            p['life'] -= 1

    def _update_laser_visuals(self):
        self.laser_visuals = [l for l in self.laser_visuals if l['life'] > 0]
        for l in self.laser_visuals:
            l['life'] -= 1

    def _manage_difficulty(self):
        if self.steps > 0 and self.steps % (10 * self.FPS) == 0:
            self.zombie_speed += 0.2
        if self.steps > 0 and self.steps % (20 * self.FPS) == 0:
            if self.zombie_base_count < self.MAX_ZOMBIES:
                self.zombie_base_count += 1

    def _spawn_zombie(self):
        edge = self.rng.integers(0, 4)
        if edge == 0: # Top
            pos = [self.rng.uniform(0, self.WIDTH), -self.ZOMBIE_RADIUS * 2]
        elif edge == 1: # Bottom
            pos = [self.rng.uniform(0, self.WIDTH), self.HEIGHT + self.ZOMBIE_RADIUS * 2]
        elif edge == 2: # Left
            pos = [-self.ZOMBIE_RADIUS * 2, self.rng.uniform(0, self.HEIGHT)]
        else: # Right
            pos = [self.WIDTH + self.ZOMBIE_RADIUS * 2, self.rng.uniform(0, self.HEIGHT)]
        
        self.zombies.append({
            'pos': np.array(pos, dtype=float),
            'anim_offset': self.rng.uniform(0, 2 * math.pi)
        })

    def _respawn_zombies(self):
        while len(self.zombies) < self.zombie_base_count:
            self._spawn_zombie()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render player base
        px, py = self.PLAYER_POS
        pygame.gfxdraw.filled_circle(self.screen, px, py, self.PLAYER_RADIUS, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, px, py, self.PLAYER_RADIUS, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_circle(self.screen, px, py, self.PLAYER_RADIUS // 2, self.COLOR_BG)
        pygame.gfxdraw.aacircle(self.screen, px, py, self.PLAYER_RADIUS // 2, self.COLOR_PLAYER)

        # Render laser visuals
        for laser in self.laser_visuals:
            alpha = int(255 * (laser['life'] / 3.0))
            color = (*self.COLOR_PROJECTILE, alpha)
            line_surf = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            pygame.draw.line(line_surf, color, laser['start'], laser['end'], 3)
            self.screen.blit(line_surf, (0,0))
            
        # Render zombies
        for zombie in self.zombies:
            zx, zy = int(zombie['pos'][0]), int(zombie['pos'][1])
            anim_y = math.sin(zombie['anim_offset']) * 2
            pygame.gfxdraw.filled_circle(self.screen, zx, int(zy + anim_y), self.ZOMBIE_RADIUS, self.COLOR_ZOMBIE)
            pygame.gfxdraw.aacircle(self.screen, zx, int(zy + anim_y), self.ZOMBIE_RADIUS, self.COLOR_ZOMBIE)

        # Render particles
        for p in self.particles:
            life_ratio = p['life'] / p['max_life']
            size = int(self.ZOMBIE_RADIUS * 0.5 * life_ratio)
            if size > 0:
                color = (*self.COLOR_PARTICLE, int(255 * life_ratio))
                part_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
                pygame.draw.circle(part_surf, color, (size, size), size)
                self.screen.blit(part_surf, (int(p['pos'][0]-size), int(p['pos'][1]-size)))

        # Render crosshair
        cx, cy = int(self.crosshair_pos[0]), int(self.crosshair_pos[1])
        size = 10
        pygame.draw.line(self.screen, self.COLOR_CROSSHAIR, (cx - size, cy), (cx + size, cy), 2)
        pygame.draw.line(self.screen, self.COLOR_CROSSHAIR, (cx, cy - size), (cx, cy + size), 2)

    def _render_ui(self):
        score_text = self.font.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        time_sec = math.ceil(max(0, self.time_remaining) / self.FPS)
        timer_text = self.font.render(f"TIME: {time_sec}", True, self.COLOR_TEXT)
        text_rect = timer_text.get_rect(topright=(self.WIDTH - 10, 10))
        self.screen.blit(timer_text, text_rect)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation.
        '''
        print("Running implementation validation...")
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")