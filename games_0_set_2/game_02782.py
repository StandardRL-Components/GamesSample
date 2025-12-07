
# Generated: 2025-08-27T21:25:19.427149
# Source Brief: brief_02782.md
# Brief Index: 2782

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: ↑↓←→ to move. Hold SPACE to mine nearby asteroids. Avoid the red enemy ships."
    )

    game_description = (
        "Mine asteroids for valuable ore while dodging enemy ships in this top-down space arcade game. Collect 100 ore to win!"
    )

    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen dimensions
        self.W, self.H = 640, 400

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.H, self.W, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.W, self.H))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("consola", 20, bold=True)
        self.font_game_over = pygame.font.SysFont("consola", 50, bold=True)

        # Colors
        self.COLOR_BG = (10, 15, 30)
        self.COLOR_PLAYER = (0, 255, 136)
        self.COLOR_PLAYER_GLOW = (0, 255, 136, 50)
        self.COLOR_ENEMY = (255, 51, 51)
        self.COLOR_ENEMY_GLOW = (255, 51, 51, 50)
        self.COLOR_ASTEROID = (128, 128, 128)
        self.COLOR_ORE = (255, 255, 0)
        self.COLOR_BEAM = (255, 255, 255, 150)
        self.COLOR_TEXT = (220, 220, 220)

        # Game parameters
        self.PLAYER_SPEED = 4.0
        self.PLAYER_RADIUS = 10
        self.ENEMY_SPEED = 1.0
        self.ENEMY_RADIUS = 8
        self.MINING_RANGE = 120
        self.MINING_RATE = 1
        self.WIN_SCORE = 100
        self.MAX_STEPS = 1000
        self.NUM_ASTEROIDS = 15
        self.NUM_ENEMIES = 5
        self.NUM_STARS = 200
        self.RISK_BONUS_DISTANCE = 75

        # Initialize state variables
        self.player_pos = None
        self.player_vel = None
        self.enemies = []
        self.asteroids = []
        self.particles = []
        self.stars = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_over_message = ""
        self.mining_target = None
        
        self.reset()
        # self.validate_implementation() # Optional validation call

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_over_message = ""
        self.mining_target = None
        
        self.player_pos = np.array([self.W / 2, self.H / 2], dtype=float)
        self.player_vel = np.array([0, 0], dtype=float)

        self.particles.clear()
        
        # Create a static starfield
        self.stars = [
            (
                self.np_random.integers(0, self.W),
                self.np_random.integers(0, self.H),
                self.np_random.uniform(0.5, 1.5),
            )
            for _ in range(self.NUM_STARS)
        ]

        # Create asteroids
        self.asteroids = []
        for _ in range(self.NUM_ASTEROIDS):
            self._create_asteroid()

        # Create enemies
        self.enemies = []
        for _ in range(self.NUM_ENEMIES):
            patrol_center = np.array([
                self.np_random.uniform(50, self.W - 50),
                self.np_random.uniform(50, self.H - 50)
            ])
            self.enemies.append({
                "pos": patrol_center.copy(),
                "patrol_center": patrol_center,
                "patrol_radius": self.np_random.uniform(40, 100),
                "angle": self.np_random.uniform(0, 2 * math.pi),
                "speed": self.np_random.uniform(0.8, 1.2) * self.ENEMY_SPEED
            })

        return self._get_observation(), self._get_info()

    def _create_asteroid(self):
        pos = np.array([
            self.np_random.uniform(0, self.W),
            self.np_random.uniform(0, self.H)
        ])
        # Ensure asteroids don't spawn too close to the center
        while np.linalg.norm(pos - np.array([self.W/2, self.H/2])) < 100:
            pos = np.array([
                self.np_random.uniform(0, self.W),
                self.np_random.uniform(0, self.H)
            ])
        
        max_ore = self.np_random.integers(20, 50)
        num_vertices = self.np_random.integers(7, 12)
        base_radius = max_ore * 0.4
        shape = []
        for i in range(num_vertices):
            angle = 2 * math.pi * i / num_vertices
            radius = self.np_random.uniform(0.8, 1.2) * base_radius
            shape.append((math.cos(angle) * radius, math.sin(angle) * radius))
        
        self.asteroids.append({
            "pos": pos,
            "max_ore": max_ore,
            "ore": max_ore,
            "shape": shape
        })

    def step(self, action):
        if self.auto_advance:
            self.clock.tick(30)
            
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        space_held = action[1] == 1
        
        reward = -0.01 # Small time penalty
        mined_this_step = False

        # 1. Update Player
        self._update_player(movement)

        # 2. Update Enemies
        self._update_enemies()

        # 3. Mining Action
        if space_held:
            reward_bonus, mined = self._handle_mining()
            reward += reward_bonus
            if mined:
                mined_this_step = True
        else:
            self.mining_target = None
        
        if not mined_this_step:
            reward -= 0.02 # Penalty for not mining

        # 4. Update Game Objects
        self._update_particles()
        self.asteroids = [a for a in self.asteroids if a["ore"] > 0]
        if len(self.asteroids) < self.NUM_ASTEROIDS:
            self._create_asteroid()
            
        # 5. Check Termination Conditions
        terminated = False
        if self.score >= self.WIN_SCORE:
            reward += 100
            terminated = True
            self.game_over_message = "YOU WIN!"
        
        if self._check_collisions():
            reward -= 50
            terminated = True
            self.game_over_message = "GAME OVER"
            self._create_explosion(self.player_pos, self.COLOR_PLAYER, 50)
            # sfx: player_explosion

        self.steps += 1
        if self.steps >= self.MAX_STEPS and not terminated:
            terminated = True
            self.game_over_message = "TIME UP"

        self.game_over = terminated
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_player(self, movement):
        target_vel = np.array([0, 0], dtype=float)
        if movement == 1: target_vel[1] = -1
        elif movement == 2: target_vel[1] = 1
        elif movement == 3: target_vel[0] = -1
        elif movement == 4: target_vel[0] = 1
        
        if np.linalg.norm(target_vel) > 0:
            target_vel = target_vel / np.linalg.norm(target_vel) * self.PLAYER_SPEED
            # Create thruster particles
            if self.steps % 2 == 0:
                offset = -target_vel * 2
                p_vel = -target_vel * 0.5 + self.np_random.uniform(-0.5, 0.5, 2)
                self._create_particle(self.player_pos + offset, p_vel, (200, 200, 255), 15, 20)

        self.player_vel = self.player_vel * 0.8 + target_vel * 0.2
        self.player_pos += self.player_vel
        self._wrap_position(self.player_pos)

    def _update_enemies(self):
        for enemy in self.enemies:
            enemy["angle"] += enemy["speed"] * 0.05
            offset_x = math.cos(enemy["angle"]) * enemy["patrol_radius"]
            offset_y = math.sin(enemy["angle"]) * enemy["patrol_radius"]
            enemy["pos"] = enemy["patrol_center"] + np.array([offset_x, offset_y])

    def _handle_mining(self):
        reward = 0
        mined = False
        
        # Find closest asteroid
        closest_asteroid = None
        min_dist = float('inf')
        for asteroid in self.asteroids:
            dist = np.linalg.norm(self.player_pos - asteroid["pos"])
            if dist < min_dist:
                min_dist = dist
                closest_asteroid = asteroid
        
        self.mining_target = None
        if closest_asteroid and min_dist <= self.MINING_RANGE:
            self.mining_target = closest_asteroid
            # sfx: mining_laser_loop
            
            # Mine ore
            mined_amount = min(self.MINING_RATE, closest_asteroid["ore"])
            closest_asteroid["ore"] -= mined_amount
            self.score += mined_amount
            reward += mined_amount
            mined = True

            # Create ore particles
            for _ in range(int(mined_amount)):
                self._create_particle(
                    closest_asteroid["pos"],
                    (self.player_pos - closest_asteroid["pos"]) * 0.02 + self.np_random.uniform(-1, 1, 2),
                    self.COLOR_ORE, 40, 60
                )
            
            # Risk/reward bonus
            for enemy in self.enemies:
                if np.linalg.norm(self.player_pos - enemy["pos"]) < self.RISK_BONUS_DISTANCE:
                    reward += 5
                    # sfx: risk_bonus
                    break
        return reward, mined

    def _check_collisions(self):
        for enemy in self.enemies:
            if np.linalg.norm(self.player_pos - enemy["pos"]) < self.PLAYER_RADIUS + self.ENEMY_RADIUS:
                return True
        return False

    def _wrap_position(self, pos):
        pos[0] %= self.W
        pos[1] %= self.H

    def _create_particle(self, pos, vel, color, min_life, max_life):
        self.particles.append({
            "pos": pos.copy(),
            "vel": vel.copy(),
            "color": color,
            "lifespan": self.np_random.integers(min_life, max_life),
            "life": 0
        })

    def _create_explosion(self, pos, color, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 5)
            vel = np.array([math.cos(angle), math.sin(angle)]) * speed
            p_color = (
                min(255, color[0] + self.np_random.integers(-20, 20)),
                min(255, color[1] + self.np_random.integers(-20, 20)),
                min(255, color[2] + self.np_random.integers(-20, 20)),
            )
            self._create_particle(pos, vel, p_color, 20, 40)

    def _update_particles(self):
        for p in self.particles:
            p["pos"] += p["vel"]
            p["life"] += 1
        self.particles = [p for p in self.particles if p["life"] < p["lifespan"]]

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw stars
        for x, y, size in self.stars:
            pygame.draw.circle(self.screen, (200, 200, 255), (x, y), size * 0.5)

        # Draw mining beam
        if self.mining_target:
            start = self.player_pos.astype(int)
            end = self.mining_target["pos"].astype(int)
            pygame.draw.line(self.screen, self.COLOR_BEAM, start, end, 2)
            
        # Draw asteroids
        for a in self.asteroids:
            scale = (a["ore"] / a["max_ore"]) ** 0.5
            if scale > 0.1:
                points = [(p[0] * scale + a["pos"][0], p[1] * scale + a["pos"][1]) for p in a["shape"]]
                pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_ASTEROID)
                pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_ASTEROID)

        # Draw enemies
        for e in self.enemies:
            pos_int = e["pos"].astype(int)
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], self.ENEMY_RADIUS+2, self.COLOR_ENEMY_GLOW)
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], self.ENEMY_RADIUS+2, self.COLOR_ENEMY_GLOW)
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], self.ENEMY_RADIUS, self.COLOR_ENEMY)
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], self.ENEMY_RADIUS, self.COLOR_ENEMY)

        # Draw particles
        for p in self.particles:
            alpha = 1.0 - (p["life"] / p["lifespan"])
            size = (2.5 * alpha)
            if size > 0.5:
                pygame.draw.circle(self.screen, p["color"], p["pos"], size)

        # Draw player
        if not (self.game_over and self._check_collisions()):
            pos_int = self.player_pos.astype(int)
            angle = math.atan2(self.player_vel[1], self.player_vel[0]) if np.linalg.norm(self.player_vel) > 0.1 else -math.pi/2
            
            # Glow
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], self.PLAYER_RADIUS+4, self.COLOR_PLAYER_GLOW)
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], self.PLAYER_RADIUS+4, self.COLOR_PLAYER_GLOW)

            # Ship body
            p1 = (pos_int[0] + math.cos(angle) * self.PLAYER_RADIUS, pos_int[1] + math.sin(angle) * self.PLAYER_RADIUS)
            p2 = (pos_int[0] + math.cos(angle + 2.2) * self.PLAYER_RADIUS, pos_int[1] + math.sin(angle + 2.2) * self.PLAYER_RADIUS)
            p3 = (pos_int[0] + math.cos(angle - 2.2) * self.PLAYER_RADIUS, pos_int[1] + math.sin(angle - 2.2) * self.PLAYER_RADIUS)
            
            pygame.gfxdraw.aapolygon(self.screen, (p1, p2, p3), self.COLOR_PLAYER)
            pygame.gfxdraw.filled_polygon(self.screen, (p1, p2, p3), self.COLOR_PLAYER)

    def _render_ui(self):
        # Draw score
        score_text = self.font_ui.render(f"ORE: {self.score}/{self.WIN_SCORE}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Draw steps
        steps_text = self.font_ui.render(f"STEPS: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT)
        self.screen.blit(steps_text, (self.W - steps_text.get_width() - 10, 10))

        # Draw game over message
        if self.game_over:
            msg_surf = self.font_game_over.render(self.game_over_message, True, self.COLOR_TEXT)
            msg_rect = msg_surf.get_rect(center=(self.W / 2, self.H / 2))
            self.screen.blit(msg_surf, msg_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_pos": self.player_pos.tolist(),
            "num_asteroids": len(self.asteroids),
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        print("✓ Validating implementation...")
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.H, self.W, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (self.H, self.W, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.H, self.W, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example usage:
if __name__ == "__main__":
    env = GameEnv(render_mode="rgb_array")
    env.validate_implementation()
    
    # To run the game with manual controls:
    # This requires a display.
    try:
        import os
        os.environ["SDL_VIDEODRIVER"] = "x11" # Or "windows", "dummy", etc.
        
        screen = pygame.display.set_mode((env.W, env.H))
        pygame.display.set_caption("Asteroid Miner")
        
        obs, info = env.reset()
        done = False
        
        while not done:
            movement = 0 # none
            space = 0 # released
            shift = 0 # released
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True

            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
            
            if keys[pygame.K_SPACE]: space = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
                
            action = [movement, space, shift]
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated or truncated:
                print(f"Game Over! Final Score: {info['score']}, Steps: {info['steps']}")
                obs, info = env.reset() # Auto-reset after a game over

            # Transpose obs for pygame display
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()

    except ImportError:
        print("Pygame not fully available for display. Skipping manual play.")
    except pygame.error as e:
        print(f"Pygame display error: {e}. Skipping manual play.")
        print("This might be because you are in a headless environment.")
        print("The environment is still valid for training.")

    env.close()