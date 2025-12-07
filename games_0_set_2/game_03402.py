
# Generated: 2025-08-27T23:14:51.218965
# Source Brief: brief_03402.md
# Brief Index: 3402

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from typing import List, Tuple, Dict, Any
import os
import pygame


class GameEnv(gym.Env):
    """
    Gymnasium environment for a top-down arcade space mining game.
    The player pilots a ship, collects ore from asteroids, and avoids collisions.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # User-facing control string
    user_guide = (
        "Controls: Arrow keys to move. Hold Space near an asteroid to mine."
    )

    # User-facing description of the game
    game_description = (
        "Pilot a mining ship through a dangerous asteroid field. "
        "Extract ore to meet your quota, but watch out for collisions that damage your ship."
    )

    # Frames auto-advance at 30fps for smooth graphics
    auto_advance = True

    # --- Game Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    FPS = 30
    MAX_STEPS = 1500  # Increased for more exploration time

    # Player settings
    PLAYER_SIZE = 12
    PLAYER_ACCELERATION = 0.5
    PLAYER_FRICTION = 0.95
    PLAYER_MAX_SPEED = 5
    PLAYER_MAX_HEALTH = 3
    PLAYER_INVULNERABILITY_FRAMES = 60

    # Asteroid settings
    ASTEROID_COUNT = 10
    ASTEROID_MIN_SIZE = 15
    ASTEROID_MAX_SIZE = 35
    ASTEROID_MIN_SPEED = 0.2
    ASTEROID_MAX_SPEED = 0.8
    ASTEROID_MIN_ORE = 2
    ASTEROID_MAX_ORE = 8
    ASTEROID_VERTICES = 12

    # Mining settings
    MINING_DISTANCE = 50
    MINING_RATE = 5  # Ore per second

    # Goal
    ORE_GOAL = 50

    # Colors
    COLOR_BG = (10, 15, 30)
    COLOR_PLAYER = (0, 255, 128)
    COLOR_PLAYER_SHIELD = (0, 255, 128, 64)
    COLOR_ASTEROID = (120, 120, 120)
    COLOR_ORE = (255, 220, 0)
    COLOR_TEXT = (220, 220, 255)
    COLOR_HEALTH_BAR_FG = (0, 255, 128)
    COLOR_HEALTH_BAR_BG = (60, 0, 0)
    COLOR_HEALTH_BAR_WARN = (255, 255, 0)
    COLOR_HEALTH_BAR_DANGER = (255, 0, 0)
    COLOR_MINING_BEAM = (100, 200, 255, 150)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup for headless rendering
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_floating = pygame.font.SysFont("monospace", 14, bold=True)

        # Game state variables are initialized in reset()
        self.player_pos: np.ndarray = None
        self.player_vel: np.ndarray = None
        self.player_health: int = 0
        self.player_invulnerable_timer: int = 0
        self.ore_collected: int = 0
        self.asteroids: List[Dict[str, Any]] = []
        self.particles: List[Dict[str, Any]] = []
        self.floating_texts: List[Dict[str, Any]] = []
        self.stars: List[Tuple[int, int, int]] = []
        self.steps: int = 0
        self.score: float = 0
        self.mining_target_idx: int = -1
        self.mining_progress: float = 0.0

        self.reset()
        self.validate_implementation()


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.ore_collected = 0
        self.player_health = self.PLAYER_MAX_HEALTH
        self.player_invulnerable_timer = 0
        
        self.player_pos = np.array([self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2], dtype=np.float32)
        self.player_vel = np.array([0.0, 0.0], dtype=np.float32)

        self.asteroids = []
        self.particles = []
        self.floating_texts = []
        
        # Generate a static starfield for the episode
        self.stars = []
        for _ in range(150):
            self.stars.append(
                (
                    self.np_random.integers(0, self.SCREEN_WIDTH),
                    self.np_random.integers(0, self.SCREEN_HEIGHT),
                    self.np_random.integers(1, 4) # size/brightness
                )
            )

        # Generate initial asteroids, ensuring they don't spawn on the player
        for _ in range(self.ASTEROID_COUNT):
            while True:
                pos = np.array([
                    self.np_random.uniform(0, self.SCREEN_WIDTH),
                    self.np_random.uniform(0, self.SCREEN_HEIGHT)
                ])
                if np.linalg.norm(pos - self.player_pos) > self.ASTEROID_MAX_SIZE * 2:
                    self._create_asteroid(position=pos)
                    break
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = -0.02  # Small penalty for each step to encourage efficiency

        # --- Update game logic ---
        self._handle_input(action)
        self._update_player()
        self._update_asteroids()
        reward += self._handle_mining(action)
        reward += self._handle_collisions()
        self._update_effects()

        self.steps += 1
        self.score += reward

        # --- Check for termination ---
        terminated = False
        if self.ore_collected >= self.ORE_GOAL:
            reward += 100
            terminated = True
            self._create_floating_text("QUOTA MET!", self.player_pos, (0, 255, 0), 120)
        elif self.player_health <= 0:
            reward += -100
            terminated = True
            self._create_explosion(self.player_pos, 100, (255, 100, 0))
        elif self.steps >= self.MAX_STEPS:
            terminated = True
        
        if self.auto_advance:
            self.clock.tick(self.FPS)

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated is always False
            self._get_info(),
        )

    def _handle_input(self, action):
        movement, space_held, _ = action
        
        accel_vec = np.array([0.0, 0.0])
        if movement == 1:  # Up
            accel_vec[1] = -self.PLAYER_ACCELERATION
        elif movement == 2:  # Down
            accel_vec[1] = self.PLAYER_ACCELERATION
        elif movement == 3:  # Left
            accel_vec[0] = -self.PLAYER_ACCELERATION
        elif movement == 4:  # Right
            accel_vec[0] = self.PLAYER_ACCELERATION

        self.player_vel += accel_vec
        
        # Add engine trail particles if moving
        if np.linalg.norm(accel_vec) > 0:
            # sfx: player_engine_hum
            angle = math.atan2(accel_vec[1], accel_vec[0]) + math.pi
            for _ in range(2):
                p_angle = angle + self.np_random.uniform(-0.3, 0.3)
                p_vel = np.array([math.cos(p_angle), math.sin(p_angle)]) * self.np_random.uniform(1, 3)
                self._create_particle(
                    self.player_pos - accel_vec * 10, p_vel, 15, (100, 100, 255), (50, 50, 150)
                )

    def _update_player(self):
        # Apply friction
        self.player_vel *= self.PLAYER_FRICTION
        
        # Clamp speed
        speed = np.linalg.norm(self.player_vel)
        if speed > self.PLAYER_MAX_SPEED:
            self.player_vel = self.player_vel / speed * self.PLAYER_MAX_SPEED
            
        # Update position
        self.player_pos += self.player_vel
        
        # Screen wrapping
        self.player_pos[0] %= self.SCREEN_WIDTH
        self.player_pos[1] %= self.SCREEN_HEIGHT
        
        if self.player_invulnerable_timer > 0:
            self.player_invulnerable_timer -= 1

    def _update_asteroids(self):
        for asteroid in self.asteroids:
            asteroid["pos"] += asteroid["vel"]
            asteroid["angle"] += asteroid["rot_speed"]
            
            # Screen wrapping
            asteroid["pos"][0] %= self.SCREEN_WIDTH
            asteroid["pos"][1] %= self.SCREEN_HEIGHT

    def _handle_mining(self, action):
        _, space_held, _ = action
        reward = 0
        
        if not space_held:
            self.mining_target_idx = -1
            self.mining_progress = 0
            return 0

        # Find closest mineable asteroid
        if self.mining_target_idx == -1:
            min_dist = self.MINING_DISTANCE + 1
            for i, asteroid in enumerate(self.asteroids):
                dist = np.linalg.norm(self.player_pos - asteroid["pos"])
                if dist < min_dist:
                    min_dist = dist
                    self.mining_target_idx = i
        
        # If we have a target, mine it
        if self.mining_target_idx != -1:
            target = self.asteroids[self.mining_target_idx]
            dist = np.linalg.norm(self.player_pos - target["pos"])
            
            if dist > self.MINING_DISTANCE or target["ore"] <= 0:
                self.mining_target_idx = -1
                self.mining_progress = 0
                return 0
            
            # sfx: mining_beam_active
            self.mining_progress += self.MINING_RATE / self.FPS
            if self.mining_progress >= 1.0:
                self.mining_progress -= 1.0
                
                ore_mined = 1
                target["ore"] -= ore_mined
                self.ore_collected = min(self.ORE_GOAL, self.ore_collected + ore_mined)
                
                reward += 0.1 * ore_mined # Continuous reward
                
                # sfx: ore_collect_blip
                # Create ore particle effect
                for _ in range(3):
                    angle = self.np_random.uniform(0, 2 * math.pi)
                    vel = np.array([math.cos(angle), math.sin(angle)]) * self.np_random.uniform(2, 4)
                    self._create_particle(target["pos"], vel, 30, self.COLOR_ORE, (150, 120, 0))

                if target["ore"] <= 0:
                    reward += 1.0 # Event-based reward for depleting an asteroid
                    self._create_floating_text(f"+{target['initial_ore']} Ore", target["pos"], self.COLOR_ORE, 60)
                    self._reset_asteroid(self.mining_target_idx)
                    self.mining_target_idx = -1
        
        return reward

    def _handle_collisions(self):
        if self.player_invulnerable_timer > 0:
            return 0
        
        reward = 0
        for asteroid in self.asteroids:
            dist = np.linalg.norm(self.player_pos - asteroid["pos"])
            if dist < self.PLAYER_SIZE + asteroid["size"]:
                # sfx: player_hit
                self.player_health -= 1
                assert self.player_health >= 0
                self.player_invulnerable_timer = self.PLAYER_INVULNERABILITY_FRAMES
                reward -= 1.0
                
                self._create_explosion(self.player_pos, 20, (255, 150, 0))
                self._create_floating_text("-1 HP", self.player_pos, self.COLOR_HEALTH_BAR_DANGER, 60)
                
                # Pushback
                push_vec = (self.player_pos - asteroid["pos"]) / dist
                self.player_vel += push_vec * 3
                asteroid["vel"] -= push_vec * 0.5
                break # Only one collision per frame
                
        return reward

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_stars()
        self._render_particles()
        self._render_asteroids()
        if self.player_health > 0:
            self._render_player()
        self._render_floating_texts()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "health": self.player_health,
            "ore": self.ore_collected,
        }

    # --- Rendering Methods ---
    def _render_stars(self):
        for x, y, size in self.stars:
            c = max(40, 80 - size * 10)
            pygame.draw.rect(self.screen, (c, c, c+10), (x, y, size, size))

    def _render_player(self):
        # Draw mining beam
        if self.mining_target_idx != -1:
            target_pos = self.asteroids[self.mining_target_idx]["pos"]
            pygame.draw.aaline(self.screen, self.COLOR_MINING_BEAM, self.player_pos, target_pos, 2)
            
        # Draw player ship (triangle)
        angle = math.atan2(self.player_vel[1], self.player_vel[0]) if np.linalg.norm(self.player_vel) > 0.1 else -math.pi/2
        points = []
        for i in range(3):
            a = angle + i * 2 * math.pi / 3
            size = self.PLAYER_SIZE if i == 0 else self.PLAYER_SIZE * 0.7
            points.append(
                (
                    self.player_pos[0] + math.cos(a) * size,
                    self.player_pos[1] + math.sin(a) * size
                )
            )
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)

        # Draw invulnerability shield
        if self.player_invulnerable_timer > 0:
            alpha = (self.player_invulnerable_timer / self.PLAYER_INVULNERABILITY_FRAMES) * 128
            color = (*self.COLOR_PLAYER_SHIELD[:3], int(alpha))
            radius = int(self.PLAYER_SIZE * 1.5)
            temp_surf = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
            pygame.gfxdraw.filled_circle(temp_surf, radius, radius, radius, color)
            self.screen.blit(temp_surf, (int(self.player_pos[0] - radius), int(self.player_pos[1] - radius)))

    def _render_asteroids(self):
        for asteroid in self.asteroids:
            points = []
            for i in range(self.ASTEROID_VERTICES):
                a = asteroid["angle"] + i * 2 * math.pi / self.ASTEROID_VERTICES
                r = asteroid["shape"][i]
                points.append(
                    (
                        asteroid["pos"][0] + math.cos(a) * r,
                        asteroid["pos"][1] + math.sin(a) * r,
                    )
                )
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_ASTEROID)
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_ASTEROID)

    def _render_particles(self):
        for p in self.particles:
            progress = p['life'] / p['max_life']
            color = tuple(int(s + (e - s) * (1 - progress)) for s, e in zip(p['start_color'], p['end_color']))
            size = int(p['size'] * progress)
            if size > 0:
                pygame.draw.circle(self.screen, color, p['pos'].astype(int), size)

    def _render_floating_texts(self):
        for ft in self.floating_texts:
            progress = ft['life'] / ft['max_life']
            alpha = int(255 * math.sin(progress * math.pi)) # Fade in and out
            text_surf = self.font_floating.render(ft['text'], True, ft['color'])
            text_surf.set_alpha(alpha)
            pos = ft['pos'] - np.array([text_surf.get_width() / 2, text_surf.get_height() / 2])
            self.screen.blit(text_surf, pos.astype(int))

    def _render_ui(self):
        # Ore display
        ore_text = self.font_ui.render(f"ORE: {self.ore_collected}/{self.ORE_GOAL}", True, self.COLOR_TEXT)
        self.screen.blit(ore_text, (10, 10))

        # Health bar
        bar_width, bar_height = 120, 15
        bar_x, bar_y = self.SCREEN_WIDTH - bar_width - 10, 10
        health_ratio = self.player_health / self.PLAYER_MAX_HEALTH
        
        if health_ratio > 0.5:
            health_color = self.COLOR_HEALTH_BAR_FG
        elif health_ratio > 0.25:
            health_color = self.COLOR_HEALTH_BAR_WARN
        else:
            health_color = self.COLOR_HEALTH_BAR_DANGER
            
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (bar_x, bar_y, bar_width, bar_height))
        if health_ratio > 0:
            pygame.draw.rect(self.screen, health_color, (bar_x, bar_y, int(bar_width * health_ratio), bar_height))

    # --- Helper Methods for Game Logic ---
    def _create_asteroid(self, position=None, size=None):
        if position is None:
            position = np.array([
                self.np_random.uniform(0, self.SCREEN_WIDTH),
                self.np_random.uniform(0, self.SCREEN_HEIGHT)
            ])
        
        asteroid_size = size or self.np_random.uniform(self.ASTEROID_MIN_SIZE, self.ASTEROID_MAX_SIZE)
        speed = self.np_random.uniform(self.ASTEROID_MIN_SPEED, self.ASTEROID_MAX_SPEED)
        angle = self.np_random.uniform(0, 2 * math.pi)
        
        shape = [
            asteroid_size + self.np_random.uniform(-asteroid_size * 0.3, asteroid_size * 0.3)
            for _ in range(self.ASTEROID_VERTICES)
        ]
        
        ore = self.np_random.integers(self.ASTEROID_MIN_ORE, self.ASTEROID_MAX_ORE + 1)
        
        self.asteroids.append({
            "pos": position.astype(np.float32),
            "vel": np.array([math.cos(angle) * speed, math.sin(angle) * speed], dtype=np.float32),
            "size": asteroid_size,
            "angle": 0.0,
            "rot_speed": self.np_random.uniform(-0.02, 0.02),
            "shape": shape,
            "ore": ore,
            "initial_ore": ore
        })

    def _reset_asteroid(self, index):
        # Place new asteroid on the opposite side of the screen from the player
        angle_to_player = math.atan2(
            self.player_pos[1] - self.SCREEN_HEIGHT / 2,
            self.player_pos[0] - self.SCREEN_WIDTH / 2
        )
        spawn_angle = angle_to_player + math.pi
        spawn_dist = max(self.SCREEN_WIDTH, self.SCREEN_HEIGHT) / 2
        
        pos_x = self.SCREEN_WIDTH / 2 + math.cos(spawn_angle) * spawn_dist
        pos_y = self.SCREEN_HEIGHT / 2 + math.sin(spawn_angle) * spawn_dist
        
        self.asteroids.pop(index)
        self._create_asteroid(position=np.array([pos_x, pos_y]))

    def _create_particle(self, pos, vel, life, start_color, end_color, size=3):
        self.particles.append({
            "pos": pos.copy(),
            "vel": vel.copy(),
            "life": life,
            "max_life": life,
            "start_color": start_color,
            "end_color": end_color,
            "size": size
        })
    
    def _create_explosion(self, pos, count, color):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 6)
            vel = np.array([math.cos(angle), math.sin(angle)]) * speed
            life = self.np_random.integers(20, 50)
            self._create_particle(pos, vel, life, color, (50, 50, 50), self.np_random.integers(2, 5))

    def _create_floating_text(self, text, pos, color, life):
        self.floating_texts.append({
            "text": text,
            "pos": pos.copy(),
            "life": life,
            "max_life": life,
            "color": color
        })

    def _update_effects(self):
        # Update particles
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['vel'] *= 0.98
            p['life'] -= 1

        # Update floating texts
        self.floating_texts = [ft for ft in self.floating_texts if ft['life'] > 0]
        for ft in self.floating_texts:
            ft['pos'][1] -= 0.5
            ft['life'] -= 1

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
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
        assert trunc is False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    # This block allows you to play the game manually
    # Requires pygame to be installed with display support
    try:
        import os
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        env = GameEnv(render_mode="rgb_array")
        os.environ["SDL_VIDEODRIVER"] = ""
    except pygame.error:
        print("Could not create a dummy display, manual play might not work.")
        print("Continuing with headless environment for validation.")
        env = GameEnv(render_mode="rgb_array")

    # --- Manual Play Loop ---
    try:
        pygame.display.set_caption("Space Miner")
        real_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
        
        obs, info = env.reset()
        terminated = False
        
        print("\n" + "="*30)
        print("MANUAL PLAY MODE")
        print(GameEnv.user_guide)
        print("Press ESC to quit.")
        print("="*30 + "\n")

        while not terminated:
            movement = 0 # none
            space = 0
            shift = 0
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    terminated = True

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
                space = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
                shift = 1
                
            action = [movement, space, shift]
            
            obs, reward, term, trunc, info = env.step(action)
            terminated = terminated or term
            
            # Display the observation from the environment
            frame = np.transpose(obs, (1, 0, 2))
            surf = pygame.surfarray.make_surface(frame)
            real_screen.blit(surf, (0, 0))
            pygame.display.flip()

            if reward != 0:
                print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']:.2f}, Ore: {info['ore']}, Health: {info['health']}")

        print("\nGame Over!")
        print(f"Final Score: {info['score']:.2f}, Ore Collected: {info['ore']}/{GameEnv.ORE_GOAL}")

    except Exception as e:
        print(f"\nCould not start manual play mode. Pygame display might not be available.")
        print(f"Error: {e}")
        print("The environment is still valid for training in a headless setup.")

    finally:
        env.close()