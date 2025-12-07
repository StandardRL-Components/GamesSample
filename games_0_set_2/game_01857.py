
# Generated: 2025-08-28T02:55:27.605439
# Source Brief: brief_01857.md
# Brief Index: 1857

        
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
        "Controls: Arrow keys to select hop direction. Hold Shift for a short hop, or Space for a long hop."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Hop your spaceship through a dangerous asteroid field. Reach the finish line at the top before time runs out. Risky hops near asteroids grant bonus points."
    )

    # Should frames auto-advance or wait for user input?
    # This is a turn-based game, so we only advance on action.
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen dimensions
        self.W, self.H = 640, 400
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.H, self.W, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.W, self.H))
        self.clock = pygame.time.Clock()
        self.font_s = pygame.font.Font(None, 24)
        self.font_l = pygame.font.Font(None, 48)

        # Game constants
        self.MAX_STEPS = 600 # 60 seconds at 10 steps/sec (1 step = 1 hop)
        self.PLAYER_RADIUS = 10
        self.FINISH_LINE_Y = 30
        self.RISKY_HOP_DISTANCE = 50
        
        # Colors
        self.COLOR_BG = (10, 10, 25)
        self.COLOR_PLAYER = (50, 255, 50)
        self.COLOR_OBSTACLE = (255, 50, 50)
        self.COLOR_FINISH = (255, 255, 0)
        self.COLOR_UI = (220, 220, 220)
        self.COLOR_PARTICLE = (255, 255, 255)
        self.COLOR_EXPLOSION = (255, 150, 0)
        
        # State variables - these will be initialized in reset()
        self.player_pos = None
        self.asteroids = []
        self.particles = []
        self.stars = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        self.obstacle_speed = 0.0
        self.np_random = None

        self.reset()
        # self.validate_implementation() # Optional: call to check compliance

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.player_pos = pygame.Vector2(self.W / 2, self.H - 40)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        self.obstacle_speed = 1.0
        
        self.asteroids = []
        for _ in range(15):
            self.asteroids.append(self._create_asteroid(on_screen=True))
            
        self.particles = []
        
        self.stars = []
        for _ in range(150):
            self.stars.append({
                "pos": pygame.Vector2(self.np_random.uniform(0, self.W), self.np_random.uniform(0, self.H)),
                "size": self.np_random.uniform(0.5, 1.5),
                "speed_mult": self.np_random.uniform(0.1, 0.4)
            })

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1  # Boolean for long hop
        shift_held = action[2] == 1  # Boolean for short hop
        
        self.steps += 1
        reward = -0.02  # Base cost per hop
        
        # 1. Player Movement
        if movement > 0:
            hop_distance = 15 if space_held else (2 if shift_held else 5)
            hop_vector = pygame.Vector2(0, 0)
            if movement == 1: hop_vector.y = -1 # Up
            elif movement == 2: hop_vector.y = 1 # Down
            elif movement == 3: hop_vector.x = -1 # Left
            elif movement == 4: hop_vector.x = 1 # Right

            self.player_pos += hop_vector * hop_distance
            self._create_hop_particles(hop_vector)

            # World wrapping
            self.player_pos.x %= self.W
            if self.player_pos.y > self.H: self.player_pos.y = 0
            elif self.player_pos.y < 0: self.player_pos.y = self.H
            
            # Risky hop reward
            is_risky = False
            for ast in self.asteroids:
                dist = self.player_pos.distance_to(ast["pos"])
                if ast["radius"] < dist < self.RISKY_HOP_DISTANCE:
                    is_risky = True
                    break
            if is_risky:
                reward += 0.1

        # 2. Update Game State
        self._update_asteroids()
        self._update_particles()
        self._update_stars()
        
        # Difficulty scaling
        if self.steps > 0 and self.steps % 100 == 0:
            self.obstacle_speed += 0.1

        # 3. Check Termination Conditions
        terminated = self._check_collisions()
        if not terminated:
            terminated = self._check_win_condition()
        
        if self.steps >= self.MAX_STEPS and not terminated:
            terminated = True
            reward -= 5 # Penalty for running out of time
            
        self.game_over = terminated
        
        # 4. Calculate final reward on termination
        if self.game_over:
            if self.game_won:
                time_bonus = (self.MAX_STEPS - self.steps) * 0.05
                reward += 10 + time_bonus
            else: # Collision
                reward = -10

        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _check_collisions(self):
        for ast in self.asteroids:
            dist = self.player_pos.distance_to(ast["pos"])
            if dist < self.PLAYER_RADIUS + ast["radius"]:
                # sfx: explosion
                self._create_explosion(self.player_pos)
                self.game_won = False
                return True
        return False

    def _check_win_condition(self):
        if self.player_pos.y <= self.FINISH_LINE_Y:
            # sfx: win_sound
            self.game_won = True
            return True
        return False

    def _update_asteroids(self):
        for ast in self.asteroids[:]:
            ast["pos"].y += self.obstacle_speed
            if ast["pos"].y - ast["radius"] > self.H:
                self.asteroids.remove(ast)
                self.asteroids.append(self._create_asteroid(on_screen=False))

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"] += p["vel"]
            p["life"] -= 1
            if p["life"] <= 0:
                self.particles.remove(p)

    def _update_stars(self):
        for star in self.stars:
            star["pos"].y += self.obstacle_speed * star["speed_mult"]
            if star["pos"].y > self.H:
                star["pos"].y = 0
                star["pos"].x = self.np_random.uniform(0, self.W)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render stars
        for star in self.stars:
            alpha = int(star["speed_mult"] * 255)
            color = (alpha, alpha, alpha)
            pygame.draw.circle(self.screen, color, star["pos"], star["size"])

        # Render finish line
        for x in range(0, self.W, 20):
            pygame.draw.line(self.screen, self.COLOR_FINISH, (x, self.FINISH_LINE_Y), (x + 10, self.FINISH_LINE_Y), 2)

        # Render particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p["life"] / p["max_life"]))))
            color = (p["color"][0], p["color"][1], p["color"][2], alpha)
            
            temp_surf = pygame.Surface((p["radius"]*2, p["radius"]*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p["radius"], p["radius"]), p["radius"])
            self.screen.blit(temp_surf, (int(p["pos"].x - p["radius"]), int(p["pos"].y - p["radius"])))

        # Render asteroids
        for ast in self.asteroids:
            points = [(p + ast["pos"]) for p in ast["points"]]
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_OBSTACLE)
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_OBSTACLE)

        # Render player if not crashed
        if not (self.game_over and not self.game_won):
            # Glow effect
            for i in range(4):
                alpha = 60 - i * 15
                pygame.gfxdraw.filled_circle(self.screen, int(self.player_pos.x), int(self.player_pos.y), self.PLAYER_RADIUS + i*2, (*self.COLOR_PLAYER, alpha))
            
            # Main ship body
            pygame.gfxdraw.aacircle(self.screen, int(self.player_pos.x), int(self.player_pos.y), self.PLAYER_RADIUS, self.COLOR_PLAYER)
            pygame.gfxdraw.filled_circle(self.screen, int(self.player_pos.x), int(self.player_pos.y), self.PLAYER_RADIUS, self.COLOR_PLAYER)
            
            # Cockpit
            pygame.gfxdraw.filled_circle(self.screen, int(self.player_pos.x), int(self.player_pos.y), self.PLAYER_RADIUS // 2, (200, 255, 200))

    def _render_ui(self):
        # Time
        time_left = max(0, (self.MAX_STEPS - self.steps) / 10.0)
        time_text = self.font_s.render(f"TIME: {time_left:.1f}", True, self.COLOR_UI)
        self.screen.blit(time_text, (10, 10))
        
        # Score
        score_text = self.font_s.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI)
        score_rect = score_text.get_rect(topright=(self.W - 10, 10))
        self.screen.blit(score_text, score_rect)
        
        # Game Over message
        if self.game_over:
            msg = "YOU REACHED THE FINISH!" if self.game_won else "GAME OVER"
            color = self.COLOR_FINISH if self.game_won else self.COLOR_OBSTACLE
            end_text = self.font_l.render(msg, True, color)
            end_rect = end_text.get_rect(center=(self.W / 2, self.H / 2))
            self.screen.blit(end_text, end_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
        }

    def _create_asteroid(self, on_screen=True):
        radius = self.np_random.uniform(10, 25)
        if on_screen:
            y_pos = self.np_random.uniform(self.FINISH_LINE_Y + 50, self.H - 100)
        else: # Spawn off-screen at the top
            y_pos = -radius - self.np_random.uniform(20, 100)
        
        pos = pygame.Vector2(self.np_random.uniform(0, self.W), y_pos)

        # Procedural shape
        num_points = self.np_random.integers(7, 12)
        points = []
        for i in range(num_points):
            angle = (2 * math.pi / num_points) * i
            dist = self.np_random.uniform(radius * 0.8, radius * 1.2)
            points.append(pygame.Vector2(math.cos(angle) * dist, math.sin(angle) * dist))
            
        return {"pos": pos, "radius": radius, "points": points}

    def _create_hop_particles(self, hop_vector):
        # sfx: hop_whoosh
        for _ in range(10):
            vel = -hop_vector * self.np_random.uniform(1, 3) + pygame.Vector2(self.np_random.uniform(-0.5, 0.5), self.np_random.uniform(-0.5, 0.5))
            self.particles.append({
                "pos": self.player_pos.copy(),
                "vel": vel,
                "radius": self.np_random.uniform(1, 3),
                "life": self.np_random.integers(10, 20),
                "max_life": 20,
                "color": self.COLOR_PARTICLE
            })

    def _create_explosion(self, position):
        for _ in range(50):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 6)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                "pos": position.copy(),
                "vel": vel,
                "radius": self.np_random.uniform(2, 5),
                "life": self.np_random.integers(20, 40),
                "max_life": 40,
                "color": self.COLOR_EXPLOSION
            })

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
        assert test_obs.shape == (self.H, self.W, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.H, self.W, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.H, self.W, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    print("="*50)
    print("Arcade Hopper Environment")
    print(env.game_description)
    print(env.user_guide)
    print("="*50)

    # For human play, we need a different render mode
    # For this file, we'll just simulate and print info
    # To play, you would need to set up a pygame window and event loop
    
    # Simulate a few random steps
    for i in range(100):
        if done:
            print(f"Episode finished after {info['steps']} steps with score {info['score']:.2f}")
            obs, info = env.reset()
        
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        print(f"Step {info['steps']}: Action={action}, Reward={reward:.2f}, Score={info['score']:.2f}, Done={done}")

    env.close()