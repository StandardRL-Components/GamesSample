
# Generated: 2025-08-28T02:42:42.221782
# Source Brief: brief_01787.md
# Brief Index: 1787

        
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

    user_guide = (
        "Controls: ↑↓←→ to aim your hop. Press space to jump. Avoid hitting asteroids and running out of fuel."
    )

    game_description = (
        "Hop between asteroids to reach the goal before your fuel runs out in this top-down arcade game."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen and world dimensions
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
        self.font_ui = pygame.font.Font(None, 24)
        self.font_msg = pygame.font.Font(None, 50)

        # Colors
        self.COLOR_BG = (10, 15, 30)
        self.COLOR_PLAYER = (255, 80, 80)
        self.COLOR_PLAYER_GLOW = (255, 80, 80, 50)
        self.COLOR_ASTEROID = (120, 130, 150)
        self.COLOR_GOAL = (80, 255, 120)
        self.COLOR_GOAL_GLOW = (80, 255, 120, 70)
        self.COLOR_FUEL = (80, 120, 255)
        self.COLOR_TRAJECTORY = (255, 255, 255, 150)
        self.COLOR_TEXT = (220, 220, 240)

        # Game constants
        self.MAX_STEPS = 1500
        self.MAX_FUEL = 100.0
        self.NUM_ASTEROIDS = 15
        self.MIN_ASTEROID_RADIUS = 15
        self.MAX_ASTEROID_RADIUS = 35
        self.PLAYER_RADIUS = 8
        self.AIM_POWER_PER_STEP = 3.0
        self.MAX_AIM_POWER = 200
        self.HOP_SPEED = 5.0 # pixels per step
        self.FUEL_COST_PER_DISTANCE = 0.2

        # Initialize state variables
        self.np_random = None
        self.asteroids = []
        self.stars = []
        self.particles = []
        self.player_pos = pygame.math.Vector2(0, 0)
        self.player_vel = pygame.math.Vector2(0, 0)
        self.player_angle = 0.0
        self.player_state = "AIMING" # "AIMING", "HOPPING"
        self.aim_vector = pygame.math.Vector2(0, 0)
        self.current_asteroid_idx = -1
        self.start_asteroid_idx = -1
        self.goal_asteroid_idx = -1
        self.hop_start_pos = pygame.math.Vector2(0, 0)
        self.hop_path_vector = pygame.math.Vector2(0, 0)
        self.hop_duration = 0
        self.hop_progress = 0
        self.fuel = 0.0
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.game_outcome = "" # "VICTORY", "CRASHED", "LOST_IN_SPACE", "NO_FUEL", "TIME_UP"
        self.space_was_held = False
        
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        else:
            # Use a default generator if no seed is provided
            if self.np_random is None:
                self.np_random = np.random.default_rng()

        self.steps = 0
        self.score = 0
        self.fuel = self.MAX_FUEL
        self.game_over = False
        self.game_outcome = ""
        self.player_state = "AIMING"
        self.aim_vector.update(0, 0)
        self.space_was_held = False
        self.particles.clear()

        self._generate_level()
        self._generate_stars()

        self.current_asteroid_idx = self.start_asteroid_idx
        self.player_pos = self.asteroids[self.current_asteroid_idx]['pos'].copy()

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0.0
        
        if not self.game_over:
            # Unpack factorized action
            movement = action[0]
            space_pressed = action[1] == 1 and not self.space_was_held
            self.space_was_held = action[1] == 1

            if self.player_state == "AIMING":
                self._handle_aiming(movement, space_pressed)
            elif self.player_state == "HOPPING":
                hop_event = self._handle_hopping()
                if hop_event:
                    reward += self._process_hop_event(hop_event)

            self.fuel = max(0, self.fuel)
            if self.fuel <= 0 and self.player_state == "AIMING":
                self.game_over = True
                self.game_outcome = "NO FUEL"
                reward -= 100.0
                
        self._update_particles()
        self.steps += 1
        
        if not self.game_over and self.steps >= self.MAX_STEPS:
            self.game_over = True
            self.game_outcome = "TIME UP"
            reward -= 100.0

        terminated = self.game_over
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_aiming(self, movement, space_pressed):
        # Update aim vector
        if movement == 1: self.aim_vector.y -= self.AIM_POWER_PER_STEP
        elif movement == 2: self.aim_vector.y += self.AIM_POWER_PER_STEP
        elif movement == 3: self.aim_vector.x -= self.AIM_POWER_PER_STEP
        elif movement == 4: self.aim_vector.x += self.AIM_POWER_PER_STEP

        if self.aim_vector.length() > self.MAX_AIM_POWER:
            self.aim_vector.scale_to_length(self.MAX_AIM_POWER)
            
        if self.aim_vector.length_squared() > 0:
            self.player_angle = self.aim_vector.angle_to(pygame.math.Vector2(0, -1))

        # Initiate hop
        if space_pressed and self.aim_vector.length() > 5:
            self._initiate_hop()

    def _initiate_hop(self):
        # Calculate hop path considering world wrap
        target_pos = self.player_pos + self.aim_vector
        dx = target_pos.x - self.player_pos.x
        dy = target_pos.y - self.player_pos.y
        if abs(dx) > self.W / 2: dx = -math.copysign(self.W - abs(dx), dx)
        if abs(dy) > self.H / 2: dy = -math.copysign(self.H - abs(dy), dy)
        
        self.hop_path_vector.update(dx, dy)
        distance = self.hop_path_vector.length()
        fuel_cost = distance * self.FUEL_COST_PER_DISTANCE

        if self.fuel >= fuel_cost:
            # sfx: player_jump
            self.fuel -= fuel_cost
            self.score -= fuel_cost * 0.1 # Small penalty for fuel use

            self.player_state = "HOPPING"
            self.hop_start_pos = self.player_pos.copy()
            self.hop_duration = max(1, distance / self.HOP_SPEED)
            self.hop_progress = 0
            
            self._create_particles(self.player_pos, 20, self.COLOR_PLAYER, 2.5, 20)
            self.current_asteroid_idx = -1 # No longer on an asteroid
        else:
            # sfx: action_fail
            self.aim_vector.update(0, 0) # Reset aim on failed hop

    def _handle_hopping(self):
        self.hop_progress += 1.0
        
        # Update position
        t = self.hop_progress / self.hop_duration
        displacement = self.hop_path_vector * t
        self.player_pos = self.hop_start_pos + displacement
        self.player_pos.x %= self.W
        self.player_pos.y %= self.H

        # Check for mid-hop collision
        for i, ast in enumerate(self.asteroids):
            if i != self.goal_asteroid_idx: # Can't crash into goal
                if self._check_wrapped_collision(self.player_pos, self.PLAYER_RADIUS, ast['pos'], ast['radius']):
                    # sfx: explosion
                    self._create_particles(self.player_pos, 50, self.COLOR_PLAYER, 4, 40)
                    return {"type": "crash"}

        # Check for hop completion
        if self.hop_progress >= self.hop_duration:
            # sfx: player_land
            # Check for landing
            for i, ast in enumerate(self.asteroids):
                if self._check_wrapped_collision(self.player_pos, self.PLAYER_RADIUS, ast['pos'], ast['radius']):
                    if i == self.goal_asteroid_idx:
                        self._create_particles(self.player_pos, 100, self.COLOR_GOAL, 5, 60)
                        return {"type": "goal"}
                    else:
                        self._create_particles(self.player_pos, 20, self.COLOR_ASTEROID, 2, 20)
                        return {"type": "land", "asteroid_idx": i}
            
            # Lost in space
            self._create_particles(self.player_pos, 10, self.COLOR_FUEL, 1, 30)
            return {"type": "lost"}
        
        return None

    def _process_hop_event(self, event):
        self.player_state = "AIMING"
        self.aim_vector.update(0, 0)
        
        if event["type"] == "crash":
            self.game_over = True
            self.game_outcome = "CRASHED"
            return -100.0
        elif event["type"] == "goal":
            self.game_over = True
            self.game_outcome = "VICTORY"
            self.score += 100.0
            return 100.0
        elif event["type"] == "land":
            self.current_asteroid_idx = event["asteroid_idx"]
            self.player_pos = self.asteroids[self.current_asteroid_idx]['pos'].copy()
            self.score += 10.0
            return 10.0
        elif event["type"] == "lost":
            self.game_over = True
            self.game_outcome = "LOST IN SPACE"
            return -100.0
        return 0.0

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
            "fuel": self.fuel,
            "player_state": self.player_state,
        }

    def _generate_level(self):
        self.asteroids.clear()
        placed_asteroids = []
        for _ in range(self.NUM_ASTEROIDS):
            for _ in range(100): # Max 100 attempts to place
                radius = self.np_random.integers(self.MIN_ASTEROID_RADIUS, self.MAX_ASTEROID_RADIUS + 1)
                pos = pygame.math.Vector2(
                    self.np_random.integers(radius, self.W - radius),
                    self.np_random.integers(radius, self.H - radius)
                )
                
                # Check for overlap
                is_overlapping = False
                for p_pos, p_radius in placed_asteroids:
                    if pos.distance_to(p_pos) < radius + p_radius + 10: # 10px buffer
                        is_overlapping = True
                        break
                
                if not is_overlapping:
                    self.asteroids.append({'pos': pos, 'radius': radius})
                    placed_asteroids.append((pos, radius))
                    break
        
        # Designate start and goal
        indices = list(range(len(self.asteroids)))
        self.np_random.shuffle(indices)
        self.start_asteroid_idx = indices[0]
        self.goal_asteroid_idx = indices[1]

    def _generate_stars(self):
        self.stars.clear()
        for _ in range(200):
            self.stars.append({
                'pos': pygame.math.Vector2(self.np_random.random() * self.W, self.np_random.random() * self.H),
                'size': self.np_random.random() * 1.5 + 0.5,
                'brightness': self.np_random.integers(50, 150)
            })

    def _render_game(self):
        # Stars
        for star in self.stars:
            star['brightness'] += self.np_random.integers(-5, 6)
            star['brightness'] = np.clip(star['brightness'], 50, 150)
            color = (star['brightness'], star['brightness'], star['brightness'])
            pygame.draw.circle(self.screen, color, star['pos'], star['size'])

        # Asteroids
        for i, ast in enumerate(self.asteroids):
            color = self.COLOR_GOAL if i == self.goal_asteroid_idx else self.COLOR_ASTEROID
            glow_color = self.COLOR_GOAL_GLOW if i == self.goal_asteroid_idx else None
            self._draw_wrapped_object(lambda p, r: self._draw_glowing_circle(p, r, color, glow_color), ast['pos'], ast['radius'])

        # Trajectory line
        if self.player_state == "AIMING" and self.aim_vector.length() > 0:
            target = self.player_pos + self.aim_vector
            self._draw_dashed_line_wrapped(self.player_pos, target, self.COLOR_TRAJECTORY)

        # Player
        self._draw_wrapped_object(self._draw_player, self.player_pos, self.PLAYER_RADIUS * 2)

        # Particles
        self._render_particles()

    def _render_ui(self):
        # Fuel bar
        fuel_rect_bg = pygame.Rect(10, 10, 200, 20)
        pygame.draw.rect(self.screen, (40, 40, 70), fuel_rect_bg, border_radius=4)
        fuel_width = max(0, (self.fuel / self.MAX_FUEL) * 200)
        fuel_rect_fg = pygame.Rect(10, 10, fuel_width, 20)
        pygame.draw.rect(self.screen, self.COLOR_FUEL, fuel_rect_fg, border_radius=4)
        
        # Score
        score_text = self.font_ui.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.W - score_text.get_width() - 10, 10))

        # Game Over Message
        if self.game_over:
            overlay = pygame.Surface((self.W, self.H), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            msg_text = self.font_msg.render(self.game_outcome, True, self.COLOR_TEXT)
            msg_rect = msg_text.get_rect(center=(self.W / 2, self.H / 2))
            self.screen.blit(msg_text, msg_rect)
            
    def _draw_player(self, pos, _):
        # Glow
        pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y), 18, self.COLOR_PLAYER_GLOW)
        pygame.gfxdraw.aacircle(self.screen, int(pos.x), int(pos.y), 18, self.COLOR_PLAYER_GLOW)
        
        # Triangle body
        p1 = pygame.math.Vector2(0, -self.PLAYER_RADIUS * 1.5).rotate(-self.player_angle) + pos
        p2 = pygame.math.Vector2(-self.PLAYER_RADIUS, self.PLAYER_RADIUS * 0.5).rotate(-self.player_angle) + pos
        p3 = pygame.math.Vector2(self.PLAYER_RADIUS, self.PLAYER_RADIUS * 0.5).rotate(-self.player_angle) + pos
        points = [(p1.x, p1.y), (p2.x, p2.y), (p3.x, p3.y)]
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)

    def _draw_glowing_circle(self, pos, radius, color, glow_color):
        if glow_color:
            glow_radius = int(radius * 1.5)
            pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y), glow_radius, glow_color)
            pygame.gfxdraw.aacircle(self.screen, int(pos.x), int(pos.y), glow_radius, glow_color)
        pygame.gfxdraw.aacircle(self.screen, int(pos.x), int(pos.y), int(radius), color)
        pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y), int(radius), color)

    def _draw_wrapped_object(self, draw_func, pos, radius):
        for dx in [0, self.W, -self.W]:
            for dy in [0, self.H, -self.H]:
                draw_func(pos + pygame.math.Vector2(dx, dy), radius)

    def _check_wrapped_collision(self, pos1, r1, pos2, r2):
        for dx in [0, self.W, -self.W]:
            for dy in [0, self.H, -self.H]:
                p2_wrapped = pos2 + pygame.math.Vector2(dx, dy)
                if pos1.distance_to(p2_wrapped) < r1 + r2:
                    return True
        return False

    def _draw_dashed_line_wrapped(self, p1, p2, color):
        dx = p2.x - p1.x
        dy = p2.y - p1.y
        if abs(dx) > self.W / 2: dx = -math.copysign(self.W - abs(dx), dx)
        if abs(dy) > self.H / 2: dy = -math.copysign(self.H - abs(dy), dy)
        
        path_vec = pygame.math.Vector2(dx, dy)
        length = path_vec.length()
        
        num_dashes = int(length / 10)
        if num_dashes == 0: return

        for i in range(num_dashes):
            if i % 2 == 0:
                start_t = i / num_dashes
                end_t = (i + 0.5) / num_dashes
                start_pos = p1 + path_vec * start_t
                end_pos = p1 + path_vec * end_t
                start_pos.x %= self.W
                start_pos.y %= self.H
                end_pos.x %= self.W
                end_pos.y %= self.H
                pygame.draw.aaline(self.screen, color, start_pos, end_pos)
                
    def _create_particles(self, pos, count, color, speed, lifespan):
        for _ in range(count):
            angle = self.np_random.random() * 2 * math.pi
            vel = pygame.math.Vector2(math.cos(angle), math.sin(angle)) * (self.np_random.random() * speed)
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'lifespan': self.np_random.integers(lifespan / 2, lifespan),
                'color': color,
                'size': self.np_random.random() * 3 + 1
            })

    def _update_particles(self):
        for p in self.particles:
            p['pos'] += p['vel']
            p['lifespan'] -= 1
            p['vel'] *= 0.95 # Damping
        self.particles = [p for p in self.particles if p['lifespan'] > 0]

    def _render_particles(self):
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['lifespan'] / 20))))
            color = (*p['color'][:3], alpha)
            
            temp_surf = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p['size'], p['size']), p['size'])
            self.screen.blit(temp_surf, p['pos'] - pygame.math.Vector2(p['size'], p['size']), special_flags=pygame.BLEND_RGBA_ADD)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
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
    env = GameEnv()
    obs, info = env.reset()
    terminated = False
    
    # Simple keyboard mapping
    key_map = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }

    # For human play
    pygame.display.set_caption("Asteroid Hopper")
    screen = pygame.display.set_mode((env.W, env.H))
    
    movement = 0
    space = 0
    shift = 0
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                terminated = False

        keys = pygame.key.get_pressed()
        
        movement = 0
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space = 1 if keys[pygame.K_SPACE] else 0
        shift = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Display the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        env.clock.tick(30)
        
        if terminated:
            print(f"Game Over! Score: {info['score']}, Outcome: {env.game_outcome}")
            # Wait for reset
            pass

    env.close()