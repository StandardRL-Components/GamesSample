
# Generated: 2025-08-27T21:53:27.638681
# Source Brief: brief_02940.md
# Brief Index: 2940

        
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


class Particle:
    """A simple particle for visual effects."""
    def __init__(self, pos, vel, lifespan, color, size_range):
        self.pos = pygame.Vector2(pos)
        self.vel = pygame.Vector2(vel)
        self.lifespan = lifespan
        self.max_lifespan = lifespan
        self.color = color
        self.size = random.uniform(size_range[0], size_range[1])

    def update(self, dt):
        self.pos += self.vel * dt
        self.lifespan -= dt
        
    def draw(self, surface):
        if self.lifespan > 0:
            alpha = int(255 * (self.lifespan / self.max_lifespan))
            radius = int(self.size * (self.lifespan / self.max_lifespan))
            if radius > 0:
                pygame.gfxdraw.filled_circle(
                    surface, int(self.pos.x), int(self.pos.y), radius, (*self.color, alpha)
                )

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: ←→ to aim, hold Space to charge jump, release Space to leap. "
        "Reach the red planet before time runs out!"
    )

    game_description = (
        "An arcade-style space adventure. Hop between procedurally generated asteroids "
        "to reach a destination within a time limit. Precision and timing are key."
    )

    auto_advance = True
    
    # --- Constants ---
    # Colors
    COLOR_BG = (10, 10, 26)
    COLOR_PLAYER = (60, 255, 160)
    COLOR_PLAYER_GLOW = (150, 255, 200)
    COLOR_ASTEROID = (128, 128, 148)
    COLOR_ASTEROID_DARK = (100, 100, 110)
    COLOR_DESTINATION = (255, 69, 0)
    COLOR_DESTINATION_GLOW = (255, 165, 0)
    COLOR_TRAJECTORY = (0, 191, 255)
    COLOR_UI_TEXT = (240, 240, 255)
    COLOR_POWER_BAR = (0, 255, 127)
    COLOR_POWER_BAR_BG = (70, 70, 90)

    # Physics & Game Rules
    FPS = 60
    GRAVITY = 90.0
    PLAYER_RADIUS = 10
    MAX_JUMP_POWER = 450.0
    MIN_JUMP_POWER = 50.0
    JUMP_CHARGE_RATE = 250.0
    AIM_ROTATE_SPEED = 2.5  # radians per second
    TIME_PER_STAGE = 30.0
    MAX_STAGES = 5

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.width, self.height = 640, 400
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.width, self.height))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 64)
        
        self.render_mode = render_mode
        self.np_random = None

        self._initialize_state()
        self.validate_implementation()
    
    def _initialize_state(self):
        """Initializes all state variables. Called by __init__ and reset."""
        self.steps = 0
        self.score = 0
        self.stage = 0
        self.game_over = False
        self.game_won = False
        self.win_message = ""
        
        self.player_pos = pygame.Vector2(0, 0)
        self.player_vel = pygame.Vector2(0, 0)
        self.player_state = "on_asteroid"  # 'on_asteroid', 'in_air'
        self.on_asteroid_id = 0
        
        self.asteroids = []
        self.destination = {'pos': pygame.Vector2(0,0), 'radius': 0}
        
        self.timer = self.TIME_PER_STAGE
        self.aim_angle = -math.pi / 2  # Start aiming up
        self.jump_power = self.MIN_JUMP_POWER
        
        self.prev_space_held = False
        self.particles = []
        self.stars = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
        
        self._initialize_state()
        self._generate_stars()
        self._setup_stage()
        
        return self._get_observation(), self._get_info()
    
    def _setup_stage(self):
        """Generates the layout for the current stage."""
        self.asteroids.clear()
        self.particles.clear()
        self.timer = self.TIME_PER_STAGE
        self.player_vel.xy = 0, 0
        self.jump_power = self.MIN_JUMP_POWER
        
        # Difficulty scaling
        difficulty_mult = 1.0 + self.stage * 0.1
        min_dist = 100 * difficulty_mult
        max_dist = 200 * difficulty_mult
        asteroid_radius = max(10, 30 * (1 - self.stage * 0.1))

        # Create starting asteroid
        start_pos = pygame.Vector2(self.width / 2, self.height - 40)
        self.asteroids.append({'pos': start_pos, 'radius': asteroid_radius + 10, 'angle': 0, 'speed': random.uniform(-0.5, 0.5)})
        self.player_pos.xy = start_pos.xy
        self.player_state = "on_asteroid"
        self.on_asteroid_id = 0
        
        # Procedurally generate a path of asteroids
        current_pos = start_pos
        for _ in range(4):
            angle = random.uniform(-math.pi * 0.6, -math.pi * 0.4) # bias upwards
            dist = random.uniform(min_dist, max_dist)
            next_pos = current_pos + pygame.Vector2(dist, 0).rotate_rad(angle)
            
            # Clamp to screen bounds with padding
            next_pos.x = np.clip(next_pos.x, 50, self.width - 50)
            next_pos.y = np.clip(next_pos.y, 50, self.height - 50)
            
            self.asteroids.append({'pos': next_pos, 'radius': asteroid_radius, 'angle': 0, 'speed': random.uniform(-0.5, 0.5)})
            current_pos = next_pos
            
        # Create destination
        dest_pos = current_pos + pygame.Vector2(0, -random.uniform(min_dist * 0.8, max_dist * 0.8))
        dest_pos.x = np.clip(dest_pos.x, 100, self.width - 100)
        dest_pos.y = np.clip(dest_pos.y, 50, 100)
        self.destination = {'pos': dest_pos, 'radius': asteroid_radius * 1.5}

    def _generate_stars(self):
        self.stars = []
        for _ in range(150):
            self.stars.append({
                'pos': pygame.Vector2(random.randint(0, self.width), random.randint(0, self.height)),
                'size': random.choice([1, 1, 1, 2]),
                'color': random.choice([(100,100,100), (150,150,150), (200,200,255)])
            })

    def step(self, action):
        dt = self.clock.tick(self.FPS) / 1000.0
        reward = 0.0

        if not self.game_over:
            self._handle_input(action, dt)
            reward = self._update_game_state(dt)
        
        self.steps += 1
        self.score += reward
        
        terminated = self._check_termination()
        
        if terminated and not self.game_won:
            if self.timer <= 0:
                reward -= 50
                self.win_message = "TIME'S UP"
            else:
                reward -= 100
                self.win_message = "LOST IN SPACE"
            self.score += reward
            
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action, dt):
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        
        if self.player_state == 'on_asteroid':
            # Aiming
            if movement == 3: # Left
                self.aim_angle -= self.AIM_ROTATE_SPEED * dt
            if movement == 4: # Right
                self.aim_angle += self.AIM_ROTATE_SPEED * dt
                
            # Charging jump
            if space_held:
                self.jump_power = min(self.MAX_JUMP_POWER, self.jump_power + self.JUMP_CHARGE_RATE * dt)
            
            # Jumping (on release)
            if not space_held and self.prev_space_held:
                self.player_state = 'in_air'
                self.player_vel = pygame.Vector2(self.jump_power, 0).rotate_rad(self.aim_angle)
                self.jump_power = self.MIN_JUMP_POWER
                self._create_particles(self.player_pos, 30, (200, 200, 255), 200, (2, 5)) # sfx: jump
                
        self.prev_space_held = space_held

    def _update_game_state(self, dt):
        reward_this_step = 0
        self.timer = max(0, self.timer - dt)

        # Update asteroids
        for asteroid in self.asteroids:
            asteroid['angle'] += asteroid['speed'] * dt
            
        # Update particles
        self.particles = [p for p in self.particles if p.lifespan > 0]
        for p in self.particles:
            p.update(dt)

        if self.player_state == 'in_air':
            reward_this_step -= 0.01 # Penalty for being in air
            self.player_vel.y += self.GRAVITY * dt
            self.player_pos += self.player_vel * dt
            
            # Create trail
            if self.steps % 3 == 0:
                self._create_particles(self.player_pos, 1, self.COLOR_PLAYER_GLOW, 50, (1, 3), 0.5)

            # Check for landing on asteroids
            for i, asteroid in enumerate(self.asteroids):
                if self.player_pos.distance_to(asteroid['pos']) < self.PLAYER_RADIUS + asteroid['radius']:
                    self.player_state = 'on_asteroid'
                    self.on_asteroid_id = i
                    self.player_pos = asteroid['pos'] + (self.player_pos - asteroid['pos']).normalize() * asteroid['radius']
                    self.player_vel.xy = 0, 0
                    self.aim_angle = -math.pi / 2
                    reward_this_step += 5.0 # Reward for landing
                    self._create_particles(self.player_pos, 20, (220, 220, 220), 100, (1, 4)) # sfx: land
                    break
            
            # Check for landing on destination
            if self.player_pos.distance_to(self.destination['pos']) < self.PLAYER_RADIUS + self.destination['radius']:
                self.game_won = True
                self.stage += 1
                reward_this_step += 10.0 + 10.0 * (self.timer / self.TIME_PER_STAGE)
                if self.stage >= self.MAX_STAGES:
                    self.win_message = "YOU WIN!"
                    self.game_over = True
                else:
                    self.win_message = "STAGE COMPLETE!"
                    self._setup_stage() # sfx: win_stage
        
        return reward_this_step

    def _check_termination(self):
        if self.game_over:
            return True
        if self.timer <= 0:
            self.game_over = True
            return True
        if self.player_state == 'in_air':
            if not (0 < self.player_pos.x < self.width and -50 < self.player_pos.y < self.height + 50):
                self.game_over = True # sfx: fall
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
            "stage": self.stage,
            "timer": self.timer
        }

    def _render_game(self):
        # Draw stars
        for star in self.stars:
            pygame.draw.circle(self.screen, star['color'], (int(star['pos'].x), int(star['pos'].y)), star['size'])

        # Draw destination
        pulse = (math.sin(self.steps * 0.1) + 1) / 2
        dest_radius = int(self.destination['radius'])
        glow_radius = int(dest_radius + 5 + pulse * 5)
        glow_color = (*self.COLOR_DESTINATION_GLOW, int(100 + pulse * 100))
        pygame.gfxdraw.filled_circle(self.screen, int(self.destination['pos'].x), int(self.destination['pos'].y), glow_radius, glow_color)
        pygame.gfxdraw.filled_circle(self.screen, int(self.destination['pos'].x), int(self.destination['pos'].y), dest_radius, self.COLOR_DESTINATION)
        pygame.gfxdraw.aacircle(self.screen, int(self.destination['pos'].x), int(self.destination['pos'].y), dest_radius, self.COLOR_DESTINATION)

        # Draw asteroids
        for asteroid in self.asteroids:
            radius = int(asteroid['radius'])
            pos = asteroid['pos']
            pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y), radius, self.COLOR_ASTEROID_DARK)
            for i in range(5):
                offset_angle = asteroid['angle'] + i * (2 * math.pi / 5)
                offset = pygame.Vector2(radius * 0.2, 0).rotate_rad(offset_angle)
                pygame.gfxdraw.filled_circle(self.screen, int(pos.x + offset.x), int(pos.y + offset.y), int(radius * 0.85), self.COLOR_ASTEROID)
            pygame.gfxdraw.aacircle(self.screen, int(pos.x), int(pos.y), radius, self.COLOR_ASTEROID_DARK)

        # Draw particles
        for p in self.particles:
            p.draw(self.screen)
            
        # Draw jump trajectory
        if self.player_state == 'on_asteroid' and self.prev_space_held:
            start_pos = self.player_pos
            end_pos = start_pos + pygame.Vector2(self.jump_power, 0).rotate_rad(self.aim_angle) * 0.3
            self._draw_dashed_line(start_pos, end_pos, self.COLOR_TRAJECTORY)

        # Draw Player
        player_x, player_y = int(self.player_pos.x), int(self.player_pos.y)
        glow_radius = int(self.PLAYER_RADIUS * 1.8)
        glow_alpha = 50 + int(100 * (self.jump_power / self.MAX_JUMP_POWER)) if self.prev_space_held else 50
        pygame.gfxdraw.filled_circle(self.screen, player_x, player_y, glow_radius, (*self.COLOR_PLAYER_GLOW, glow_alpha))
        pygame.gfxdraw.filled_circle(self.screen, player_x, player_y, self.PLAYER_RADIUS, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, player_x, player_y, self.PLAYER_RADIUS, self.COLOR_PLAYER_GLOW)

    def _render_ui(self):
        # Stage Text
        stage_text = self.font_small.render(f"Stage: {self.stage + 1}/{self.MAX_STAGES}", True, self.COLOR_UI_TEXT)
        self.screen.blit(stage_text, (10, 10))

        # Timer Text
        timer_text = self.font_small.render(f"Time: {self.timer:.1f}", True, self.COLOR_UI_TEXT)
        self.screen.blit(timer_text, (self.width - timer_text.get_width() - 10, 10))

        # Power Bar
        if self.player_state == 'on_asteroid' and self.prev_space_held:
            power_ratio = (self.jump_power - self.MIN_JUMP_POWER) / (self.MAX_JUMP_POWER - self.MIN_JUMP_POWER)
            bar_width = 200
            bar_height = 15
            bar_x = self.width / 2 - bar_width / 2
            bar_y = self.height - 30
            
            pygame.draw.rect(self.screen, self.COLOR_POWER_BAR_BG, (bar_x, bar_y, bar_width, bar_height), border_radius=4)
            pygame.draw.rect(self.screen, self.COLOR_POWER_BAR, (bar_x, bar_y, bar_width * power_ratio, bar_height), border_radius=4)

        # Game Over / Win Message
        if self.win_message:
            msg_surf = self.font_large.render(self.win_message, True, self.COLOR_UI_TEXT)
            msg_rect = msg_surf.get_rect(center=(self.width / 2, self.height / 2))
            self.screen.blit(msg_surf, msg_rect)

    def _create_particles(self, pos, count, color, max_speed, size_range, lifespan=1.0):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(0, max_speed)
            vel = pygame.Vector2(speed, 0).rotate_rad(angle)
            self.particles.append(Particle(pos, vel, lifespan, color, size_range))

    def _draw_dashed_line(self, start_pos, end_pos, color, dash_length=10):
        dx = end_pos.x - start_pos.x
        dy = end_pos.y - start_pos.y
        distance = math.sqrt(dx*dx + dy*dy)
        if distance == 0: return
        
        for i in range(0, int(distance / dash_length), 2):
            t0 = i / (distance / dash_length)
            t1 = (i + 1) / (distance / dash_length)
            p0 = start_pos + pygame.Vector2(dx, dy) * t0
            p1 = start_pos + pygame.Vector2(dx, dy) * t1
            pygame.draw.line(self.screen, color, p0, p1, 2)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.height, self.width, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.height, self.width, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.height, self.width, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    # It's a demonstration of how to use the environment
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup Pygame window for human play
    pygame.display.set_caption("Asteroid Hopper")
    screen = pygame.display.set_mode((env.width, env.height))
    
    terminated = False
    running = True
    total_score = 0
    
    while running:
        # Map keyboard keys to the MultiDiscrete action space
        keys = pygame.key.get_pressed()
        
        movement = 0 # no-op
        if keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_score += reward
        
        # Draw the observation from the environment to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated:
            print(f"Episode finished. Score: {info['score']:.2f}, Steps: {info['steps']}")
            # In a real training loop, you would reset here.
            # For human play, we'll wait a bit then reset.
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_score = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
    env.close()