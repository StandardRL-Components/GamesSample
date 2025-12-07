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
        "Controls: Use arrow keys (↑↓←→) to move your green cell. "
        "Survive by eating smaller blue cells and avoiding larger red predators."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced survival game. Grow larger by consuming prey, but watch out for predators that hunt you down. "
        "Survive for 60 seconds to win."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_WIDTH = 32
    GRID_HEIGHT = 20
    CELL_SIZE = 20
    FPS = 30
    MAX_STEPS = 60 * FPS # 60 seconds

    # Colors
    COLOR_BG = (10, 20, 30)
    COLOR_GRID = (25, 35, 45)
    COLOR_PLAYER = (0, 255, 150)
    COLOR_PREY = (100, 150, 255)
    COLOR_PREDATOR = (255, 80, 80)
    COLOR_TEXT = (220, 220, 220)
    COLOR_TIMER = (255, 200, 0)
    
    # Game parameters
    INITIAL_PLAYER_SIZE = 8
    PREY_COUNT = 15
    PREDATOR_COUNT = 2
    INITIAL_PREDATOR_SPEED = 0.04
    PREDATOR_SPEED_INCREASE_INTERVAL = 10 * FPS # every 10 seconds
    PREDATOR_SPEED_INCREASE_AMOUNT = 0.01

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
        self.font_large = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        
        # Initialize state variables to None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.time_remaining = 0
        self.player = None
        self.prey_list = []
        self.predator_list = []
        self.particles = []
        self.size_thresholds_to_reward = []
        
        # self.reset() # reset is called by the wrapper or test harness

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.time_remaining = self.MAX_STEPS
        
        # Player
        start_pos = np.array([self.GRID_WIDTH / 2, self.GRID_HEIGHT / 2])
        self.player = {
            'grid_pos': start_pos.copy(),
            'visual_pos': start_pos.copy() * self.CELL_SIZE,
            'size': self.INITIAL_PLAYER_SIZE,
            'target_size': self.INITIAL_PLAYER_SIZE,
        }
        self.size_thresholds_to_reward = [12, 16, 20, 24]

        # Creatures
        self.prey_list = [self._spawn_creature('prey') for _ in range(self.PREY_COUNT)]
        self.predator_list = [self._spawn_creature('predator') for _ in range(self.PREDATOR_COUNT)]
        
        self.particles = []
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        # If game is over, do nothing but return the final state
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        step_reward = 0
        
        # --- 1. Handle Input ---
        movement = action[0]
        self._update_player_target(movement)
        
        # --- 2. Update Game Logic ---
        self._update_creatures()
        step_reward += self._handle_collisions()
        self._update_particles()
        self._update_difficulty()

        # --- 3. Update State ---
        self.steps += 1
        self.time_remaining -= 1
        
        # --- 4. Calculate Reward ---
        # Small penalty for each step to encourage action
        reward = -0.01 + step_reward
        self.score += reward

        # --- 5. Check Termination ---
        terminated = self._check_termination()
        if terminated:
            if self.win:
                reward += 100  # Victory bonus
                self.score += 100
            else:
                reward -= 100  # Eaten penalty
                self.score -= 100
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_player_target(self, movement):
        dx, dy = 0, 0
        if movement == 1: dy = -1  # Up
        elif movement == 2: dy = 1   # Down
        elif movement == 3: dx = -1  # Left
        elif movement == 4: dx = 1   # Right
        
        if dx != 0 or dy != 0:
            self.player['grid_pos'][0] = (self.player['grid_pos'][0] + dx) % self.GRID_WIDTH
            self.player['grid_pos'][1] = (self.player['grid_pos'][1] + dy) % self.GRID_HEIGHT

    def _update_creatures(self):
        # Interpolate player visual position and size
        target_visual_pos = self.player['grid_pos'] * self.CELL_SIZE + self.CELL_SIZE / 2
        self.player['visual_pos'] = self.player['visual_pos'] * 0.8 + target_visual_pos * 0.2
        self.player['size'] = self.player['size'] * 0.9 + self.player['target_size'] * 0.1

        # Move Predators
        for pred in self.predator_list:
            # Toroidal distance calculation
            delta = self.player['grid_pos'] - pred['grid_pos']
            if delta[0] > self.GRID_WIDTH / 2: delta[0] -= self.GRID_WIDTH
            if delta[0] < -self.GRID_WIDTH / 2: delta[0] += self.GRID_WIDTH
            if delta[1] > self.GRID_HEIGHT / 2: delta[1] -= self.GRID_HEIGHT
            if delta[1] < -self.GRID_HEIGHT / 2: delta[1] += self.GRID_HEIGHT
            
            dist = np.linalg.norm(delta)
            if dist > 0:
                direction = delta / dist
                pred['grid_pos'] += direction * pred['speed']
                pred['grid_pos'][0] %= self.GRID_WIDTH
                pred['grid_pos'][1] %= self.GRID_HEIGHT
            
            target_visual_pos = pred['grid_pos'] * self.CELL_SIZE + self.CELL_SIZE / 2
            pred['visual_pos'] = pred['visual_pos'] * 0.8 + target_visual_pos * 0.2

        # Move Prey randomly
        for p in self.prey_list:
            if self.np_random.random() < 0.05: # 5% chance to change direction
                p['dir'] = np.array([self.np_random.uniform(-1, 1), self.np_random.uniform(-1, 1)])
                if np.linalg.norm(p['dir']) > 0:
                    p['dir'] /= np.linalg.norm(p['dir'])
            
            p['grid_pos'] += p['dir'] * 0.2 # Prey are slow
            p['grid_pos'][0] %= self.GRID_WIDTH
            p['grid_pos'][1] %= self.GRID_HEIGHT
            
            target_visual_pos = p['grid_pos'] * self.CELL_SIZE + self.CELL_SIZE / 2
            p['visual_pos'] = p['visual_pos'] * 0.9 + target_visual_pos * 0.1

    def _handle_collisions(self):
        collision_reward = 0
        player_pos = self.player['visual_pos']
        player_size = self.player['size']

        surviving_prey = []
        eaten_count = 0

        # Player vs Prey
        for prey in self.prey_list:
            dist_sq = np.sum((player_pos - prey['visual_pos'])**2)
            is_colliding = dist_sq < (player_size + prey['size'])**2
            can_eat = player_size > prey['size']

            if is_colliding and can_eat:
                # Prey is eaten
                self.player['target_size'] += 0.5
                collision_reward += 1
                self._create_particles(prey['visual_pos'], self.COLOR_PREY, 10)
                eaten_count += 1

                # Check for size threshold reward
                if self.size_thresholds_to_reward and self.player['target_size'] >= self.size_thresholds_to_reward[0]:
                    collision_reward += 5
                    self.size_thresholds_to_reward.pop(0)
            else:
                # Prey survives
                surviving_prey.append(prey)
        
        self.prey_list = surviving_prey
        for _ in range(eaten_count):
            self.prey_list.append(self._spawn_creature('prey'))
        
        # Player vs Predator
        for pred in self.predator_list:
            dist_sq = np.sum((player_pos - pred['visual_pos'])**2)
            if dist_sq < (player_size + pred['size'])**2 and player_size < pred['size']:
                self.game_over = True
                self._create_particles(self.player['visual_pos'], self.COLOR_PLAYER, 50)
                break
        
        return collision_reward

    def _update_difficulty(self):
        if self.steps > 0 and self.steps % self.PREDATOR_SPEED_INCREASE_INTERVAL == 0:
            for pred in self.predator_list:
                pred['speed'] += self.PREDATOR_SPEED_INCREASE_AMOUNT
    
    def _check_termination(self):
        if self.time_remaining <= 0 and not self.game_over:
            self.win = True
            self.game_over = True
        return self.game_over

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_grid()
        self._render_particles()
        self._render_creatures()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_size": self.player['target_size'] if self.player else 0,
            "time_remaining": self.time_remaining,
        }

    def _spawn_creature(self, creature_type):
        pos = self.np_random.random(2) * np.array([self.GRID_WIDTH, self.GRID_HEIGHT])
        if creature_type == 'prey':
            return {
                'grid_pos': pos,
                'visual_pos': pos * self.CELL_SIZE,
                'size': self.np_random.uniform(3, 5),
                'dir': np.array([self.np_random.uniform(-1, 1), self.np_random.uniform(-1, 1)])
            }
        elif creature_type == 'predator':
            return {
                'grid_pos': pos,
                'visual_pos': pos * self.CELL_SIZE,
                'size': self.np_random.uniform(15, 20),
                'speed': self.INITIAL_PREDATOR_SPEED,
            }

    # --- Rendering Methods ---
    def _render_grid(self):
        for x in range(0, self.SCREEN_WIDTH, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))

    def _render_creatures(self):
        # Prey
        for p in self.prey_list:
            self._draw_glowing_circle(self.screen, p['visual_pos'], p['size'], self.COLOR_PREY, 0.3)
        # Predators
        for pred in self.predator_list:
            pulse = 1 + 0.1 * math.sin(self.steps * 0.2)
            self._draw_glowing_circle(self.screen, pred['visual_pos'], pred['size'] * pulse, self.COLOR_PREDATOR, 0.5)
        # Player
        if not (self.game_over and not self.win):
            self._draw_glowing_circle(self.screen, self.player['visual_pos'], self.player['size'], self.COLOR_PLAYER, 0.7)

    def _draw_glowing_circle(self, surface, pos, radius, color, glow_factor):
        pos_int = (int(pos[0]), int(pos[1]))
        radius = max(1, int(radius))
        
        # Glow layers
        for i in range(3, 0, -1):
            alpha = int(80 * glow_factor * (1 - i / 4))
            if alpha > 0:
                glow_radius = radius + i * 3
                s = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
                pygame.draw.circle(s, color + (alpha,), (glow_radius, glow_radius), glow_radius)
                surface.blit(s, (pos_int[0] - glow_radius, pos_int[1] - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)

        # Main circle with anti-aliasing
        pygame.gfxdraw.aacircle(surface, pos_int[0], pos_int[1], radius, color)
        pygame.gfxdraw.filled_circle(surface, pos_int[0], pos_int[1], radius, color)

    def _render_ui(self):
        # Time remaining
        time_text = f"TIME: {self.time_remaining // self.FPS:02d}"
        time_surf = self.font_large.render(time_text, True, self.COLOR_TIMER)
        self.screen.blit(time_surf, (self.SCREEN_WIDTH // 2 - time_surf.get_width() // 2, 10))

        # Player size
        size_text = f"SIZE: {self.player['target_size']:.1f}"
        size_surf = self.font_small.render(size_text, True, self.COLOR_TEXT)
        self.screen.blit(size_surf, (10, 10))
        
        # Score
        score_text = f"SCORE: {int(self.score)}"
        score_surf = self.font_small.render(score_text, True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (self.SCREEN_WIDTH - score_surf.get_width() - 10, 10))

        # Game Over / Win Message
        if self.game_over:
            message = "YOU WIN!" if self.win else "GAME OVER"
            color = self.COLOR_PLAYER if self.win else self.COLOR_PREDATOR
            msg_surf = self.font_large.render(message, True, color)
            self.screen.blit(msg_surf, (self.SCREEN_WIDTH // 2 - msg_surf.get_width() // 2, self.SCREEN_HEIGHT // 2 - msg_surf.get_height() // 2))

    # --- Particle Effects ---
    def _create_particles(self, pos, color, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            velocity = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
            self.particles.append({
                'pos': pos.copy(),
                'vel': velocity,
                'life': self.np_random.integers(15, 25),
                'color': color
            })

    def _update_particles(self):
        active_particles = []
        for p in self.particles:
            p['pos'] += p['vel']
            p['vel'] *= 0.95 # friction
            p['life'] -= 1
            if p['life'] > 0:
                active_particles.append(p)
        self.particles = active_particles
    
    def _render_particles(self):
        for p in self.particles:
            alpha = max(0, int(255 * (p['life'] / 25)))
            color_with_alpha = p['color'] + (alpha,)
            size = max(1, int(p['life'] / 8))
            rect = pygame.Rect(int(p['pos'][0] - size/2), int(p['pos'][1] - size/2), size, size)
            
            # Create a temporary surface for the particle to handle alpha blending correctly
            particle_surf = pygame.Surface((size, size), pygame.SRCALPHA)
            pygame.draw.rect(particle_surf, color_with_alpha, particle_surf.get_rect())
            self.screen.blit(particle_surf, rect)

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation.
        '''
        print("Validating implementation...")
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
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
    # This block allows you to play the game directly
    # To run, you might need to unset the dummy video driver
    # comment out: os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    # or run as: SDL_VIDEODRIVER=x11 python your_script_name.py
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Grid Survival")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        # --- Human Controls ---
        movement = 0 # No-op
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        action = [movement, 0, 0] # Space and Shift are not used

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()
                total_reward = 0

        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Total Reward: {total_reward:.2f}")
            # Optional: auto-reset after a delay
            # pygame.time.wait(2000)
            # obs, info = env.reset()
            # total_reward = 0

        # --- Rendering ---
        # The observation is already a rendered frame, so we just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(env.FPS)
        
    pygame.quit()