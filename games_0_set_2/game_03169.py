
# Generated: 2025-08-28T07:11:47.158065
# Source Brief: brief_03169.md
# Brief Index: 3169

        
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
        "Controls: Hold SPACE to jump over obstacles. Run as far as you can!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced, side-scrolling arcade game. Control Road Runner, dodge cacti, rocks, and "
        "holes in a procedurally generated desert to reach the end of the level."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen dimensions
        self.WIDTH, self.HEIGHT = 640, 400

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)

        # Colors
        self.COLOR_SKY = (135, 206, 235)
        self.COLOR_GROUND = (238, 214, 175)
        self.COLOR_PLAYER_BODY = (139, 69, 19)
        self.COLOR_PLAYER_HEAD = (65, 105, 225)
        self.COLOR_PLAYER_TAIL = (255, 69, 0)
        self.COLOR_CACTUS = (34, 139, 34)
        self.COLOR_ROCK = (139, 137, 137)
        self.COLOR_HOLE = (40, 40, 40)
        self.COLOR_TEXT = (50, 50, 50)
        self.COLOR_MESA_1 = (205, 133, 63)
        self.COLOR_MESA_2 = (188, 143, 143)

        # Game constants
        self.GROUND_Y = self.HEIGHT - 80
        self.GRAVITY = 0.8
        self.JUMP_STRENGTH = 15
        self.PLAYER_X = 100
        self.MAX_STEPS = 10000

        # Initialize state variables
        self.player_y = 0
        self.player_vy = 0
        self.on_ground = True
        self.prev_space_held = False

        self.obstacles = []
        self.particles = []
        self.parallax_bg = []
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.base_world_speed = 5.0
        self.world_speed = 5.0
        self.base_obstacle_interval = 80
        self.obstacle_interval = 80
        self.next_obstacle_spawn_step = 0
        
        self.rng = None

        self.reset()
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = np.random.default_rng()

        self.player_y = self.GROUND_Y
        self.player_vy = 0
        self.on_ground = True
        self.prev_space_held = False

        self.obstacles = []
        self.particles = []
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.world_speed = self.base_world_speed
        self.obstacle_interval = self.base_obstacle_interval
        self.next_obstacle_spawn_step = 100

        self._init_parallax()

        return self._get_observation(), self._get_info()

    def _init_parallax(self):
        self.parallax_bg = []
        for _ in range(20):
            # Far mesas
            self.parallax_bg.append({
                "x": self.rng.integers(0, self.WIDTH * 2),
                "y": self.GROUND_Y - self.rng.integers(20, 80),
                "w": self.rng.integers(80, 200),
                "h": self.rng.integers(20, 80),
                "speed_mod": 0.2,
                "color": self.COLOR_MESA_2
            })
            # Near mesas
            self.parallax_bg.append({
                "x": self.rng.integers(0, self.WIDTH * 2),
                "y": self.GROUND_Y - self.rng.integers(10, 40),
                "w": self.rng.integers(100, 250),
                "h": self.rng.integers(10, 40),
                "speed_mod": 0.5,
                "color": self.COLOR_MESA_1
            })
        self.parallax_bg.sort(key=lambda p: p["speed_mod"])

    def step(self, action):
        reward = 0
        self.game_over = self.steps >= self.MAX_STEPS

        if not self.game_over:
            # Unpack factorized action
            movement = action[0]
            space_held = action[1] == 1
            shift_held = action[2] == 1
            
            # 1. Handle Input
            jumped_this_frame = self._handle_input(space_held)
            
            # 2. Update Game State
            self._update_game_state()
            
            # 3. Check for cleared obstacles
            reward += self._check_cleared_obstacles()
            
            # 4. Handle Collisions
            collision = self._check_collisions()
            if collision:
                self.game_over = True
                reward = -10 # Collision penalty
            else:
                reward += 0.1 # Survival reward
                self.score += 1 # Score is distance
                
            if jumped_this_frame:
                if not self._is_obstacle_near():
                    reward -= 0.2 # Unnecessary jump penalty

            # 5. Spawn new obstacles
            self._spawn_obstacles()

            # 6. Update difficulty
            self._update_difficulty()
            
        self.steps += 1
        
        terminated = self.game_over
        if terminated and self.steps < self.MAX_STEPS: # Lost
            pass
        elif terminated and self.steps >= self.MAX_STEPS: # Won
            reward += 100
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _handle_input(self, space_held):
        jumped = False
        if space_held and not self.prev_space_held and self.on_ground:
            self.player_vy = self.JUMP_STRENGTH
            self.on_ground = False
            jumped = True
            # Sound effect placeholder: play_jump_sound()
        self.prev_space_held = space_held
        return jumped

    def _update_game_state(self):
        # Update player physics
        self.player_vy -= self.GRAVITY
        self.player_y -= self.player_vy
        
        if self.player_y >= self.GROUND_Y:
            if not self.on_ground:
                # Create dust on landing
                for _ in range(10):
                    self.particles.append({
                        'x': self.PLAYER_X,
                        'y': self.GROUND_Y,
                        'vx': self.rng.uniform(-2, 2),
                        'vy': self.rng.uniform(-2, 0),
                        'life': 20,
                        'color': (210, 180, 140)
                    })
                # Sound effect placeholder: play_land_sound()
            self.player_y = self.GROUND_Y
            self.player_vy = 0
            self.on_ground = True

        # Update obstacles
        for obs in self.obstacles:
            obs['x'] -= self.world_speed
        self.obstacles = [obs for obs in self.obstacles if obs['x'] + obs['w'] > 0]
        
        # Update particles
        for p in self.particles:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

        # Update parallax background
        for p in self.parallax_bg:
            p['x'] -= self.world_speed * p['speed_mod']
            if p['x'] + p['w'] < 0:
                p['x'] = self.WIDTH + self.rng.integers(0, 50)

    def _spawn_obstacles(self):
        if self.steps >= self.next_obstacle_spawn_step:
            obstacle_type = self.rng.choice(['cactus', 'rock', 'hole'])
            new_obstacle = {'x': self.WIDTH, 'cleared': False}
            
            if obstacle_type == 'cactus':
                new_obstacle.update({'type': 'cactus', 'w': 30, 'h': 60, 'y': self.GROUND_Y - 60})
            elif obstacle_type == 'rock':
                new_obstacle.update({'type': 'rock', 'w': 40, 'h': 30, 'y': self.GROUND_Y - 30})
            else: # hole
                new_obstacle.update({'type': 'hole', 'w': 60, 'h': 10, 'y': self.GROUND_Y})

            self.obstacles.append(new_obstacle)
            self.next_obstacle_spawn_step = self.steps + self.rng.integers(self.obstacle_interval, self.obstacle_interval + 20)

    def _update_difficulty(self):
        difficulty_mod = (self.steps // 200) * 0.2
        self.world_speed = self.base_world_speed + difficulty_mod
        self.obstacle_interval = max(30, self.base_obstacle_interval - (self.steps // 200) * 5)
        
    def _check_cleared_obstacles(self):
        reward = 0
        player_left_edge = self.PLAYER_X - 15
        for obs in self.obstacles:
            if not obs['cleared'] and obs['x'] + obs['w'] < player_left_edge:
                obs['cleared'] = True
                if obs['type'] != 'hole':
                    reward += 1.0
                # Sound effect placeholder: play_clear_obstacle_sound()
        return reward
        
    def _is_obstacle_near(self):
        player_x = self.PLAYER_X
        for obs in self.obstacles:
            if obs['type'] != 'hole' and player_x < obs['x'] < player_x + 300:
                return True
        return False

    def _check_collisions(self):
        player_rect = pygame.Rect(self.PLAYER_X - 15, self.player_y - 40, 30, 40)
        for obs in self.obstacles:
            if obs['type'] == 'hole':
                if self.on_ground and player_rect.left < obs['x'] + obs['w'] and player_rect.right > obs['x']:
                    return True
            else:
                obs_rect = pygame.Rect(obs['x'], obs['y'], obs['w'], obs['h'])
                if player_rect.colliderect(obs_rect):
                    # Sound effect placeholder: play_crash_sound()
                    return True
        return False

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_SKY)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def _render_game(self):
        # Ground
        pygame.draw.rect(self.screen, self.COLOR_GROUND, (0, self.GROUND_Y, self.WIDTH, self.HEIGHT - self.GROUND_Y))
        
        # Parallax Background
        for p in self.parallax_bg:
            pygame.draw.rect(self.screen, p['color'], (int(p['x']), int(p['y']), int(p['w']), int(p['h'])))
        
        # Obstacles
        for obs in self.obstacles:
            if obs['type'] == 'cactus':
                pygame.draw.rect(self.screen, self.COLOR_CACTUS, (int(obs['x']), int(obs['y']), int(obs['w']), int(obs['h'])))
                pygame.draw.rect(self.screen, self.COLOR_CACTUS, (int(obs['x']-5), int(obs['y']+10), int(obs['w']+10), 10))
            elif obs['type'] == 'rock':
                pygame.gfxdraw.filled_polygon(self.screen, [
                    (int(obs['x']), int(obs['y'] + obs['h'])),
                    (int(obs['x'] + obs['w'] / 2), int(obs['y'])),
                    (int(obs['x'] + obs['w']), int(obs['y'] + obs['h']))
                ], self.COLOR_ROCK)
            elif obs['type'] == 'hole':
                pygame.draw.ellipse(self.screen, self.COLOR_HOLE, (int(obs['x']), int(obs['y'] - obs['h']/2), int(obs['w']), int(obs['h'])))

        # Player
        self._render_player()

        # Particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['life'] / 20))))
            color = p['color'] + (alpha,)
            temp_surf = pygame.Surface((4, 4), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (2, 2), 2)
            self.screen.blit(temp_surf, (int(p['x']), int(p['y'])))

    def _render_player(self):
        x, y = int(self.PLAYER_X), int(self.player_y)
        
        # Body (ellipse)
        body_rect = pygame.Rect(x - 15, y - 35, 30, 25)
        pygame.draw.ellipse(self.screen, self.COLOR_PLAYER_BODY, body_rect)
        
        # Head (circle)
        pygame.draw.circle(self.screen, self.COLOR_PLAYER_HEAD, (x + 10, y - 30), 12)
        pygame.draw.circle(self.screen, (255,255,255), (x + 15, y - 33), 4) # Eye
        pygame.draw.circle(self.screen, (0,0,0), (x + 16, y - 33), 2) # Pupil

        # Beak (polygon)
        beak_points = [(x + 21, y - 30), (x + 35, y - 28), (x + 21, y - 26)]
        pygame.draw.polygon(self.screen, (255, 255, 0), beak_points)
        
        # Tail (polygon)
        tail_angle = math.sin(self.steps * 0.5) * 0.2
        tail_points = [
            (x - 15, y - 25),
            (x - 35, y - 40 + math.sin(tail_angle) * 5),
            (x - 30, y - 20 + math.cos(tail_angle) * 5)
        ]
        pygame.draw.polygon(self.screen, self.COLOR_PLAYER_TAIL, tail_points)
        
        # Legs (animated based on ground status)
        if self.on_ground:
            cycle = (self.steps % 20) / 20.0
            angle1 = math.sin(cycle * 2 * math.pi)
            angle2 = math.sin(cycle * 2 * math.pi + math.pi)
            
            # Leg 1
            pygame.draw.line(self.screen, self.COLOR_PLAYER_BODY, (x - 5, y - 12), (x - 5 + 10 * angle1, y - 2), 3)
            # Leg 2
            pygame.draw.line(self.screen, self.COLOR_PLAYER_BODY, (x + 5, y - 12), (x + 5 + 10 * angle2, y - 2), 3)
        else: # Jumping/Falling
            pygame.draw.line(self.screen, self.COLOR_PLAYER_BODY, (x - 5, y - 12), (x - 10, y - 5), 3)
            pygame.draw.line(self.screen, self.COLOR_PLAYER_BODY, (x + 5, y - 12), (x, y - 5), 3)

    def _render_ui(self):
        score_text = f"Distance: {self.score}"
        text_surface = self.font.render(score_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surface, (10, 10))

        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 128))
            self.screen.blit(overlay, (0, 0))
            
            result_text = "LEVEL COMPLETE!" if self.steps >= self.MAX_STEPS else "GAME OVER"
            result_surf = self.font.render(result_text, True, (255, 255, 255))
            result_rect = result_surf.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2 - 20))
            self.screen.blit(result_surf, result_rect)
            
            final_score_text = f"Final Distance: {self.score}"
            final_score_surf = self.small_font.render(final_score_text, True, (255, 255, 255))
            final_score_rect = final_score_surf.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2 + 20))
            self.screen.blit(final_score_surf, final_score_rect)

    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
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
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    running = True
    terminated = False
    
    # Override Pygame display for direct play
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Road Runner")
    clock = pygame.time.Clock()
    
    total_reward = 0
    
    while running:
        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            obs, info = env.reset()
            total_reward = 0
            terminated = False
            pygame.time.wait(2000) # Pause before restarting

        # --- Human Controls ---
        movement = 0 # No-op for movement
        space_held = 0
        shift_held = 0
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE]:
            space_held = 1
            
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Draw the observation from the environment to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30) # Corresponds to the game's internal clock/FPS
        
    env.close()