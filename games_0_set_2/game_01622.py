
# Generated: 2025-08-27T17:43:24.461295
# Source Brief: brief_01622.md
# Brief Index: 1622

        
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
    An arcade-style spaceship game where the player must mine 100 ore from
    asteroids while dodging hostile enemy ships.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move your ship. Press space to mine adjacent asteroids."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Navigate a treacherous asteroid field, mine ore for points, and dodge enemy ships. Collect 100 ore to win!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.CELL_SIZE = 20
        self.GRID_WIDTH = self.SCREEN_WIDTH // self.CELL_SIZE
        self.GRID_HEIGHT = self.SCREEN_HEIGHT // self.CELL_SIZE
        
        self.WIN_SCORE = 100
        self.MAX_STEPS = 1000
        self.INITIAL_ENEMIES = 5
        self.MIN_ASTEROID_ORE = 20
        self.ENEMY_SPAWN_INTERVAL = 150
        self.MAX_ENEMIES = 15
        
        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        try:
            self.font_ui = pygame.font.Font(pygame.font.get_default_font(), 18)
            self.font_game_over = pygame.font.Font(pygame.font.get_default_font(), 48)
        except IOError:
            self.font_ui = pygame.font.SysFont("monospace", 18)
            self.font_game_over = pygame.font.SysFont("monospace", 48)


        # --- Colors ---
        self.COLOR_BG = (15, 15, 30)
        self.COLOR_PLAYER = (50, 255, 150)
        self.COLOR_PLAYER_GLOW = (50, 255, 150, 50)
        self.COLOR_ENEMY = (255, 50, 100)
        self.COLOR_ENEMY_GLOW = (255, 50, 100, 50)
        self.COLOR_ASTEROID = (150, 150, 150)
        self.COLOR_ORE_PARTICLE = (255, 220, 50)
        self.COLOR_EXPLOSION = (255, 100, 50)
        self.COLOR_TEXT = (220, 220, 255)
        self.COLOR_BEAM = (200, 255, 255, 150)

        # --- State variables will be initialized in reset() ---
        self.player_pos = None
        self.enemies = []
        self.asteroids = []
        self.particles = []
        self.starfield = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        self.enemy_move_prob = 0.2
        self.mining_beam = None

        self.reset()
        
        # This check is critical for validating the implementation
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        self.enemy_move_prob = 0.2
        
        self.player_pos = np.array([self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2])
        
        self.enemies = []
        self.asteroids = []
        self.particles = []
        self.mining_beam = None

        self._generate_starfield()
        
        occupied_positions = {tuple(self.player_pos)}
        
        for _ in range(self.INITIAL_ENEMIES):
            self._spawn_enemy(occupied_positions)
        
        while self._get_total_ore() < self.MIN_ASTEROID_ORE * 2:
            self._spawn_asteroid(occupied_positions)

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            # If the game is over, do nothing and just return the final state
            reward = 0
            terminated = True
            return self._get_observation(), reward, terminated, False, self._get_info()
            
        reward = 0
        terminated = False
        self.mining_beam = None

        movement, space_held, _ = action
        space_pressed = space_held == 1

        # --- 1. Player Action ---
        if movement != 0:
            # Move Player
            dx, dy = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)][movement]
            new_pos = self.player_pos + np.array([dx, dy])
            if 0 <= new_pos[0] < self.GRID_WIDTH and 0 <= new_pos[1] < self.GRID_HEIGHT:
                self.player_pos = new_pos
                # Small penalty for moving, encouraging efficient pathing
                if not self._is_near_enemy(self.player_pos):
                     reward -= 0.1

        if space_pressed:
            # Mine Asteroid
            mined = False
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0: continue
                    check_pos = tuple(self.player_pos + np.array([dx, dy]))
                    for asteroid in self.asteroids:
                        if tuple(asteroid['pos']) == check_pos:
                            # Found an asteroid to mine
                            self.score += 1
                            asteroid['ore'] -= 1
                            # sound_effect: "mine_ore.wav"
                            
                            # Base reward for mining
                            reward += 1.0
                            
                            # Bonus reward for risky mining
                            if self._is_near_enemy(asteroid['pos'], distance=3):
                                reward += 5.0
                            
                            self._create_ore_particles(asteroid['pos'])
                            self.mining_beam = {'start': self.player_pos, 'end': asteroid['pos']}

                            if asteroid['ore'] <= 0:
                                self.asteroids.remove(asteroid)
                                # sound_effect: "asteroid_break.wav"
                            
                            mined = True
                            break
                    if mined: break
                if mined: break

        # --- 2. Update Game State ---
        self._update_enemies()
        self._update_particles()
        
        # --- 3. Spawning and Difficulty ---
        self.steps += 1
        
        if self.steps % self.ENEMY_SPAWN_INTERVAL == 0 and len(self.enemies) < self.MAX_ENEMIES:
            self._spawn_enemy({tuple(self.player_pos)} | {tuple(e['pos']) for e in self.enemies})

        if self._get_total_ore() < self.MIN_ASTEROID_ORE:
            self._spawn_asteroid({tuple(self.player_pos)} | {tuple(e['pos']) for e in self.enemies} | {tuple(a['pos']) for a in self.asteroids})

        if self.steps > 0 and self.steps % 200 == 0:
            self.enemy_move_prob = min(0.8, self.enemy_move_prob + 0.01)

        # --- 4. Check for Termination Conditions ---
        # Collision with enemy
        for enemy in self.enemies:
            if np.array_equal(self.player_pos, enemy['pos']):
                terminated = True
                self.game_over = True
                reward = -50.0 # Large penalty for losing
                self._create_explosion(self.player_pos)
                # sound_effect: "player_explosion.wav"
                break
        
        # Win condition
        if not terminated and self.score >= self.WIN_SCORE:
            terminated = True
            self.game_over = True
            self.game_won = True
            reward = 100.0 # Large reward for winning
        
        # Max steps reached
        if not terminated and self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
            # No specific reward change, rely on accumulated score
        
        return self._get_observation(), reward, terminated, False, self._get_info()

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
            "enemies": len(self.enemies),
            "asteroids": len(self.asteroids),
            "total_ore": self._get_total_ore()
        }

    # --- Helper and Spawning Methods ---
    def _to_pixels(self, pos):
        return (
            int(pos[0] * self.CELL_SIZE + self.CELL_SIZE / 2),
            int(pos[1] * self.CELL_SIZE + self.CELL_SIZE / 2)
        )

    def _get_random_empty_pos(self, occupied_positions, margin=0):
        while True:
            pos = (
                self.np_random.integers(margin, self.GRID_WIDTH - margin),
                self.np_random.integers(margin, self.GRID_HEIGHT - margin)
            )
            if pos not in occupied_positions:
                occupied_positions.add(pos)
                return np.array(pos)

    def _spawn_asteroid(self, occupied_positions):
        pos = self._get_random_empty_pos(occupied_positions)
        ore = self.np_random.integers(1, 6)
        num_vertices = self.np_random.integers(5, 9)
        radius = self.CELL_SIZE * 0.4
        
        # Generate a semi-random polygon shape
        points = []
        for i in range(num_vertices):
            angle = 2 * math.pi * i / num_vertices
            r = radius * self.np_random.uniform(0.7, 1.1)
            p_x = r * math.cos(angle)
            p_y = r * math.sin(angle)
            points.append((p_x, p_y))
            
        self.asteroids.append({'pos': pos, 'ore': ore, 'shape': points})

    def _spawn_enemy(self, occupied_positions):
        # Spawn enemies away from the center
        pos = self._get_random_empty_pos(occupied_positions, margin=5)
        self.enemies.append({'pos': pos})
    
    def _get_total_ore(self):
        return sum(a['ore'] for a in self.asteroids)

    def _update_enemies(self):
        for enemy in self.enemies:
            if self.np_random.random() < self.enemy_move_prob:
                direction = self.player_pos - enemy['pos']
                # Move along the axis with the greatest distance
                if abs(direction[0]) > abs(direction[1]):
                    enemy['pos'][0] += np.sign(direction[0])
                elif abs(direction[1]) > abs(direction[0]):
                    enemy['pos'][1] += np.sign(direction[1])
                elif np.any(direction): # If equidistant, move randomly
                    axis = self.np_random.choice(2)
                    enemy['pos'][axis] += np.sign(direction[axis])

    def _is_near_enemy(self, pos, distance=2):
        for enemy in self.enemies:
            dist = np.linalg.norm(pos - enemy['pos'])
            if dist <= distance:
                return True
        return False

    # --- Particle System ---
    def _generate_starfield(self):
        self.starfield = []
        for _ in range(200):
            self.starfield.append({
                'pos': (self.np_random.integers(0, self.SCREEN_WIDTH), self.np_random.integers(0, self.SCREEN_HEIGHT)),
                'radius': self.np_random.uniform(0.5, 1.5),
                'brightness': self.np_random.integers(50, 150)
            })

    def _create_ore_particles(self, from_pos):
        start_px = self._to_pixels(from_pos)
        end_px = self._to_pixels(self.player_pos)
        for _ in range(5):
            self.particles.append({
                'type': 'ore',
                'pos': np.array(start_px, dtype=float) + self.np_random.uniform(-3, 3, 2),
                'vel': (np.array(end_px) - np.array(start_px)) / 20.0 + self.np_random.uniform(-1, 1, 2),
                'radius': self.np_random.uniform(2, 4),
                'lifetime': 20
            })

    def _create_explosion(self, at_pos):
        center_px = self._to_pixels(at_pos)
        for _ in range(50):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 5)
            self.particles.append({
                'type': 'explosion',
                'pos': np.array(center_px, dtype=float),
                'vel': np.array([math.cos(angle) * speed, math.sin(angle) * speed]),
                'radius': self.np_random.uniform(3, 7),
                'lifetime': 30
            })

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['lifetime'] -= 1
            if p['type'] == 'explosion':
                p['vel'] *= 0.95 # Slow down
            if p['lifetime'] <= 0:
                self.particles.remove(p)

    # --- Rendering Methods ---
    def _render_game(self):
        # Starfield
        for star in self.starfield:
            color = (star['brightness'], star['brightness'], star['brightness'])
            pygame.draw.circle(self.screen, color, star['pos'], star['radius'])

        # Asteroids
        for asteroid in self.asteroids:
            center_px = self._to_pixels(asteroid['pos'])
            points = [(center_px[0] + p[0], center_px[1] + p[1]) for p in asteroid['shape']]
            pygame.draw.polygon(self.screen, self.COLOR_ASTEROID, points)
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_ASTEROID)

        # Mining Beam
        if self.mining_beam:
            start_px = self._to_pixels(self.mining_beam['start'])
            end_px = self._to_pixels(self.mining_beam['end'])
            pygame.draw.aaline(self.screen, self.COLOR_BEAM, start_px, end_px, 1)

        # Enemies
        for enemy in self.enemies:
            center_px = self._to_pixels(enemy['pos'])
            size = self.CELL_SIZE * 0.8
            rect = pygame.Rect(center_px[0] - size/2, center_px[1] - size/2, size, size)
            pygame.draw.rect(self.screen, self.COLOR_ENEMY, rect, border_radius=2)
            pygame.gfxdraw.aacircle(self.screen, center_px[0], center_px[1], int(size*0.7), self.COLOR_ENEMY_GLOW)

        # Player
        if not (self.game_over and not self.game_won):
            center_px = self._to_pixels(self.player_pos)
            size = self.CELL_SIZE * 0.8
            p1 = (center_px[0], center_px[1] - size * 0.6)
            p2 = (center_px[0] - size * 0.5, center_px[1] + size * 0.4)
            p3 = (center_px[0] + size * 0.5, center_px[1] + size * 0.4)
            pygame.draw.polygon(self.screen, self.COLOR_PLAYER, [p1, p2, p3])
            pygame.gfxdraw.aapolygon(self.screen, [p1, p2, p3], self.COLOR_PLAYER)
            pygame.gfxdraw.aacircle(self.screen, center_px[0], center_px[1], int(size*0.8), self.COLOR_PLAYER_GLOW)

        # Particles
        for p in self.particles:
            pos_int = (int(p['pos'][0]), int(p['pos'][1]))
            radius = int(p['radius'] * (p['lifetime'] / 30.0))
            if radius > 0:
                color = self.COLOR_ORE_PARTICLE if p['type'] == 'ore' else self.COLOR_EXPLOSION
                pygame.draw.circle(self.screen, color, pos_int, radius)

    def _render_ui(self):
        # Score display
        score_text = self.font_ui.render(f"ORE: {self.score}/{self.WIN_SCORE}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Game Over / Win message
        if self.game_over:
            message = "YOU WIN!" if self.game_won else "GAME OVER"
            color = self.COLOR_PLAYER if self.game_won else self.COLOR_ENEMY
            text_surface = self.font_game_over.render(message, True, color)
            text_rect = text_surface.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(text_surface, text_rect)

    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
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

# --- Example Usage ---
if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    
    # Use a separate Pygame window for human play
    pygame.display.init()
    human_screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Asteroid Miner")
    
    obs, info = env.reset()
    done = False
    
    print("\n" + "="*30)
    print("ASTEROID MINER")
    print(env.game_description)
    print(env.user_guide)
    print("="*30 + "\n")
    
    # Game loop for human play
    running = True
    while running:
        # --- Action mapping for human keyboard ---
        movement = 0 # no-op
        space_held = 0
        shift_held = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        elif keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
            
        if keys[pygame.K_SPACE]:
            space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_held = 1
            
        action = [movement, space_held, shift_held]
        
        # --- Step the environment ---
        obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Render to the human-facing window ---
        # The observation is already a rendered frame, we just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        human_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Steps: {info['steps']}")
            pygame.time.wait(3000) # Pause for 3 seconds
            obs, info = env.reset()

        env.clock.tick(10) # Control human play speed

    env.close()