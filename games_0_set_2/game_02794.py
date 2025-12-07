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


class Particle:
    """A simple class for managing particles for visual effects."""
    def __init__(self, x, y, vx, vy, color, radius, lifespan):
        self.pos = pygame.Vector2(x, y)
        self.vel = pygame.Vector2(vx, vy)
        self.color = color
        self.radius = radius
        self.lifespan = lifespan
        self.initial_lifespan = lifespan

    def update(self):
        self.pos += self.vel
        self.lifespan -= 1
        self.radius = max(0, self.radius * (self.lifespan / self.initial_lifespan))

    def draw(self, surface):
        if self.lifespan > 0 and self.radius > 1:
            alpha = int(255 * (self.lifespan / self.initial_lifespan))
            r, g, b = self.color
            # Use gfxdraw for anti-aliased, alpha-blended circles
            pygame.gfxdraw.filled_circle(surface, int(self.pos.x), int(self.pos.y), int(self.radius), (r, g, b, alpha))
            pygame.gfxdraw.aacircle(surface, int(self.pos.x), int(self.pos.y), int(self.radius), (r, g, b, alpha))


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to steer the snail. Hold Space for a speed boost."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A top-down snail racing game. Navigate through three stages, avoiding obstacles "
        "and managing your boost to beat the clock. Grazing obstacles gives you a time bonus!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    # Screen
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30

    # Colors
    COLOR_BG = (25, 45, 25)
    COLOR_TRACK = (60, 110, 60)
    COLOR_SNAIL = (255, 105, 180) # Hot Pink
    COLOR_SNAIL_BOOST = (255, 255, 0) # Yellow
    COLOR_SNAIL_SHELL = (220, 50, 120)
    COLOR_OBSTACLE = (200, 40, 40)
    COLOR_RISKY_ZONE = (255, 200, 0)
    COLOR_FINISH_LINE_1 = (255, 255, 255)
    COLOR_FINISH_LINE_2 = (0, 0, 0)
    COLOR_UI_TEXT = (240, 240, 240)
    COLOR_PARTICLE_TRAIL = (173, 216, 230) # Light Blue
    COLOR_PARTICLE_COLLISION = (255, 69, 0) # OrangeRed
    COLOR_PARTICLE_RISK = (255, 215, 0) # Gold

    # Game settings
    TOTAL_STAGES = 3
    TIME_PER_STAGE = 60.0  # seconds
    MAX_COLLISIONS_PER_STAGE = 3

    # Snail physics
    SNAIL_RADIUS = 12
    SNAIL_ACCELERATION = 0.4
    SNAIL_MAX_SPEED = 3.0
    SNAIL_DRAG = 0.95
    BOOST_SPEED_MULTIPLIER = 1.8
    BOOST_DURATION_MAX = 2.0 * FPS # 2 seconds of boost
    BOOST_RECHARGE_RATE = 1.5
    BOOST_COOLDOWN_FRAMES = 0.5 * FPS

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 32)
        
        self.np_random = None # Will be seeded in reset()

        self.snail_pos = pygame.Vector2(0, 0)
        self.snail_vel = pygame.Vector2(0, 0)
        self.obstacles = []
        self.risky_zones = []
        self.particles = []
        
        # self.reset() is called here to initialize state, but gym standard is to call it externally.
        # It's kept for compatibility with the original structure.
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.current_stage = 1
        self.collisions_this_stage = 0
        
        self.boost_charge = self.BOOST_DURATION_MAX
        self.boost_cooldown = 0
        self.is_boosting = False

        self.snail_pos = pygame.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT - 40)
        self.snail_vel = pygame.Vector2(0, 0)
        
        self.particles.clear()
        
        self._setup_stage()

        return self._get_observation(), self._get_info()

    def _setup_stage(self):
        self.time_remaining = self.TIME_PER_STAGE * self.FPS
        self.collisions_this_stage = 0
        self.obstacles.clear()
        self.risky_zones.clear()

        stage_configs = {
            1: {'count': 3, 'speed': 0.5, 'radius': 20},
            2: {'count': 4, 'speed': 0.7, 'radius': 25},
            3: {'count': 5, 'speed': 0.9, 'radius': 30},
        }
        config = stage_configs[self.current_stage]
        
        for _ in range(config['count']):
            # Ensure obstacles are not too close to start or finish
            pos = pygame.Vector2(
                self.np_random.uniform(50, self.SCREEN_WIDTH - 50),
                self.np_random.uniform(80, self.SCREEN_HEIGHT - 120)
            )
            
            # Ensure no overlap with existing obstacles
            while any(pos.distance_to(obs['pos']) < config['radius'] * 2 + 20 for obs in self.obstacles):
                 pos = pygame.Vector2(
                    self.np_random.uniform(50, self.SCREEN_WIDTH - 50),
                    self.np_random.uniform(80, self.SCREEN_HEIGHT - 120)
                )

            pattern = self.np_random.choice(['linear_h', 'linear_v', 'circular'])
            obstacle = {
                'pos': pos,
                'radius': config['radius'],
                'speed': config['speed'],
                'pattern': pattern,
                'angle': self.np_random.uniform(0, 2 * math.pi),
                'center': pos.copy(),
                'path_radius': self.np_random.uniform(20, 50)
            }
            self.obstacles.append(obstacle)
            self.risky_zones.append({
                'pos': obstacle['pos'],
                'radius': obstacle['radius'] * 2,
                'obstacle_radius': obstacle['radius'],
                'traversed': False
            })

    def step(self, action):
        if self.auto_advance:
            self.clock.tick(self.FPS)

        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        
        reward = 0
        dist_before_move = self.snail_pos.y

        self._handle_input(movement, space_held)
        self._update_snail()
        self._update_obstacles()
        self._update_particles()
        
        reward += self._check_collisions_and_risks()
        
        # Continuous reward for moving towards finish line (upwards)
        dist_after_move = self.snail_pos.y
        reward += (dist_before_move - dist_after_move) * 0.01

        self.time_remaining -= 1
        self.steps += 1
        
        # Check for stage completion
        if self.snail_pos.y < 30: # Finish line area
            reward += 10
            self.current_stage += 1
            if self.current_stage > self.TOTAL_STAGES:
                self.game_over = True
                reward += 100 # Victory bonus
            else:
                self._setup_stage()
        
        terminated = self._check_termination()
        if terminated and not self.game_over: # Lost the game
            reward = -100 # Timeout or too many collisions
            self.game_over = True
        
        truncated = False # This environment does not truncate
        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, movement, space_held):
        acc = pygame.Vector2(0, 0)
        if movement == 1: acc.y -= 1
        if movement == 2: acc.y += 1
        if movement == 3: acc.x -= 1
        if movement == 4: acc.x += 1
        if acc.length() > 0:
            acc.scale_to_length(self.SNAIL_ACCELERATION)
        self.snail_vel += acc

        # Handle boost
        self.boost_cooldown = max(0, self.boost_cooldown - 1)
        if space_held and self.boost_charge > 0 and self.boost_cooldown == 0:
            self.is_boosting = True
            self.boost_charge -= 1
        else:
            self.is_boosting = False
        
        if not self.is_boosting:
            self.boost_charge = min(self.BOOST_DURATION_MAX, self.boost_charge + self.BOOST_RECHARGE_RATE)
        
        if self.is_boosting and self.boost_charge <= 0:
            self.is_boosting = False
            self.boost_cooldown = self.BOOST_COOLDOWN_FRAMES


    def _update_snail(self):
        # Apply drag
        self.snail_vel *= self.SNAIL_DRAG

        # Cap speed
        current_max_speed = self.SNAIL_MAX_SPEED
        if self.is_boosting:
            current_max_speed *= self.BOOST_SPEED_MULTIPLIER
        
        if self.snail_vel.length() > current_max_speed:
            self.snail_vel.scale_to_length(current_max_speed)

        # Update position
        self.snail_pos += self.snail_vel

        # Boundary checks
        if self.snail_pos.x < self.SNAIL_RADIUS:
            self.snail_pos.x = self.SNAIL_RADIUS
            self.snail_vel.x *= -0.5
        if self.snail_pos.x > self.SCREEN_WIDTH - self.SNAIL_RADIUS:
            self.snail_pos.x = self.SCREEN_WIDTH - self.SNAIL_RADIUS
            self.snail_vel.x *= -0.5
        if self.snail_pos.y < self.SNAIL_RADIUS:
            self.snail_pos.y = self.SNAIL_RADIUS
            self.snail_vel.y *= -0.5
        if self.snail_pos.y > self.SCREEN_HEIGHT - self.SNAIL_RADIUS:
            self.snail_pos.y = self.SCREEN_HEIGHT - self.SNAIL_RADIUS
            self.snail_vel.y *= -0.5
            
        # Add slime trail particles
        if self.snail_vel.length() > 0.5:
            p_vel = -self.snail_vel * 0.1
            p = Particle(self.snail_pos.x, self.snail_pos.y, p_vel.x, p_vel.y, self.COLOR_PARTICLE_TRAIL, self.SNAIL_RADIUS*0.5, self.FPS)
            self.particles.append(p)
    
    def _update_obstacles(self):
        for obs in self.obstacles:
            obs['angle'] += 0.02 * obs['speed']
            if obs['pattern'] == 'circular':
                obs['pos'].x = obs['center'].x + math.cos(obs['angle']) * obs['path_radius']
                obs['pos'].y = obs['center'].y + math.sin(obs['angle']) * obs['path_radius']
            elif obs['pattern'] == 'linear_h':
                obs['pos'].x = obs['center'].x + math.sin(obs['angle']) * obs['path_radius']
            elif obs['pattern'] == 'linear_v':
                obs['pos'].y = obs['center'].y + math.sin(obs['angle']) * obs['path_radius']
        
        # Update risky zone positions to match obstacles
        for i, zone in enumerate(self.risky_zones):
            zone['pos'] = self.obstacles[i]['pos']


    def _update_particles(self):
        self.particles = [p for p in self.particles if p.lifespan > 0]
        for p in self.particles:
            p.update()

    def _check_collisions_and_risks(self):
        reward = 0
        # Check obstacle collisions
        for obs in self.obstacles:
            if self.snail_pos.distance_to(obs['pos']) < self.SNAIL_RADIUS + obs['radius']:
                reward -= 5
                self.collisions_this_stage += 1
                # Bounce effect
                self.snail_vel *= -0.8
                # Collision particles
                for _ in range(15):
                    angle = self.np_random.uniform(0, 2*math.pi)
                    speed = self.np_random.uniform(1, 4)
                    p_vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
                    p = Particle(self.snail_pos.x, self.snail_pos.y, p_vel.x, p_vel.y, self.COLOR_PARTICLE_COLLISION, 4, self.FPS/2)
                    self.particles.append(p)
                # To prevent multiple collision detections for one event
                self.snail_pos += self.snail_vel.normalize() * (self.SNAIL_RADIUS + obs['radius'])

        # Check risky zones
        for zone in self.risky_zones:
            dist = self.snail_pos.distance_to(zone['pos'])
            is_in_risk_zone = dist < zone['radius'] and dist > zone['obstacle_radius']
            
            if is_in_risk_zone and not zone['traversed']:
                reward += 5
                zone['traversed'] = True # Grant reward only once per pass
                # Risky particles
                for _ in range(10):
                    angle = self.np_random.uniform(0, 2*math.pi)
                    speed = self.np_random.uniform(0.5, 2)
                    p_vel = (self.snail_pos - zone['pos']).normalize().rotate(self.np_random.uniform(-90, 90)) * speed
                    p = Particle(self.snail_pos.x, self.snail_pos.y, p_vel.x, p_vel.y, self.COLOR_PARTICLE_RISK, 3, self.FPS/3)
                    self.particles.append(p)

            elif not is_in_risk_zone:
                zone['traversed'] = False # Reset when snail leaves the zone

        return reward

    def _check_termination(self):
        if self.time_remaining <= 0:
            return True
        if self.collisions_this_stage >= self.MAX_COLLISIONS_PER_STAGE:
            return True
        if self.game_over: # e.g. from winning
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw track
        pygame.draw.rect(self.screen, self.COLOR_TRACK, (0, 30, self.SCREEN_WIDTH, self.SCREEN_HEIGHT - 60))

        # Draw finish line
        for i in range(0, self.SCREEN_WIDTH, 20):
            color = self.COLOR_FINISH_LINE_1 if (i // 20) % 2 == 0 else self.COLOR_FINISH_LINE_2
            pygame.draw.rect(self.screen, color, (i, 10, 20, 20))

        # Draw particles
        for p in self.particles:
            p.draw(self.screen)
        
        # Draw risky zones (under obstacles)
        for zone in self.risky_zones:
            pygame.gfxdraw.filled_circle(self.screen, int(zone['pos'].x), int(zone['pos'].y), int(zone['radius']), (*self.COLOR_RISKY_ZONE, 50))
            pygame.gfxdraw.aacircle(self.screen, int(zone['pos'].x), int(zone['pos'].y), int(zone['radius']), (*self.COLOR_RISKY_ZONE, 80))

        # Draw obstacles
        for obs in self.obstacles:
            pygame.gfxdraw.filled_circle(self.screen, int(obs['pos'].x), int(obs['pos'].y), obs['radius'], self.COLOR_OBSTACLE)
            pygame.gfxdraw.aacircle(self.screen, int(obs['pos'].x), int(obs['pos'].y), obs['radius'], self.COLOR_OBSTACLE)

        # Draw snail
        snail_color = self.COLOR_SNAIL_BOOST if self.is_boosting else self.COLOR_SNAIL
        pos = (int(self.snail_pos.x), int(self.snail_pos.y))
        # Shell
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.SNAIL_RADIUS, self.COLOR_SNAIL_SHELL)
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.SNAIL_RADIUS, self.COLOR_SNAIL_SHELL)
        # Body
        body_radius = int(self.SNAIL_RADIUS * 0.8)
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], body_radius, snail_color)
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], body_radius, snail_color)
        # Head for direction
        if self.snail_vel.length() > 0.1:
            head_pos = self.snail_pos + self.snail_vel.normalize() * self.SNAIL_RADIUS * 0.7
            pygame.draw.circle(self.screen, (255,255,255), (int(head_pos.x), int(head_pos.y)), 4)
            pygame.draw.circle(self.screen, (0,0,0), (int(head_pos.x), int(head_pos.y)), 2)

    def _render_ui(self):
        # Stage display
        stage_text = self.font_small.render(f"Stage: {self.current_stage}/{self.TOTAL_STAGES}", True, self.COLOR_UI_TEXT)
        self.screen.blit(stage_text, (10, 10))

        # Time display
        time_str = f"Time: {max(0, self.time_remaining / self.FPS):.1f}"
        time_text = self.font_small.render(time_str, True, self.COLOR_UI_TEXT)
        self.screen.blit(time_text, (self.SCREEN_WIDTH - time_text.get_width() - 10, 10))

        # Score display
        score_text = self.font_small.render(f"Score: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH/2 - score_text.get_width()/2, self.SCREEN_HEIGHT - 35))

        # Collision display
        collision_text = self.font_small.render(f"Hits: {self.collisions_this_stage}/{self.MAX_COLLISIONS_PER_STAGE}", True, self.COLOR_UI_TEXT)
        self.screen.blit(collision_text, (10, self.SCREEN_HEIGHT - 35))
        
        # Boost bar
        boost_pct = self.boost_charge / self.BOOST_DURATION_MAX
        bar_width = 150
        bar_height = 15
        bar_x = self.SCREEN_WIDTH - bar_width - 10
        bar_y = self.SCREEN_HEIGHT - bar_height - 15
        pygame.draw.rect(self.screen, (50, 50, 50), (bar_x, bar_y, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_SNAIL_BOOST, (bar_x, bar_y, bar_width * boost_pct, bar_height))
        boost_text = pygame.font.Font(None, 20).render("BOOST", True, self.COLOR_UI_TEXT)
        self.screen.blit(boost_text, (bar_x - boost_text.get_width() - 5, bar_y))
        
        # Game Over / Victory text
        if self.game_over:
            if self.current_stage > self.TOTAL_STAGES:
                msg = "VICTORY!"
            else:
                msg = "GAME OVER"
            end_text = self.font_large.render(msg, True, self.COLOR_UI_TEXT)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            # Draw a semi-transparent background for the text
            s = pygame.Surface(text_rect.inflate(20,20).size, pygame.SRCALPHA)
            s.fill((0,0,0,150))
            self.screen.blit(s, text_rect.inflate(20,20).topleft)
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "stage": self.current_stage,
            "time_remaining": self.time_remaining / self.FPS,
            "collisions": self.collisions_this_stage,
        }

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually.
    # To do so, you might need to comment out the line:
    # os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    # at the top of the file, as it prevents a display window from opening.
    
    # Re-enable video driver for manual play
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup Pygame window for human play
    pygame.display.set_caption("Snail Racer")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    running = True
    total_reward = 0
    
    while running:
        # --- Human Controls ---
        movement = 0 # no-op
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # --- Rendering ---
        # The observation is already a rendered frame, so we just need to display it.
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Steps: {info['steps']}")
            # Wait a bit before resetting
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0

        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

    env.close()