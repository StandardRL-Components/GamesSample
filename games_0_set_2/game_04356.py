
# Generated: 2025-08-28T02:09:11.693549
# Source Brief: brief_04356.md
# Brief Index: 4356

        
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
        "Controls: Press space to jump. You have three jumps before you must land."
    )

    game_description = (
        "A side-scrolling arcade game where you pilot a hopping spaceship through fields of obstacles."
    )

    auto_advance = True

    # --- Constants ---
    # Colors
    COLOR_BG = (20, 10, 40)
    COLOR_PLAYER = (0, 192, 255)
    COLOR_PLAYER_GLOW = (0, 100, 200)
    COLOR_OBSTACLE = (255, 0, 128)
    COLOR_OBSTACLE_GLOW = (200, 0, 100)
    COLOR_GOAL = (0, 255, 128)
    COLOR_TEXT = (255, 255, 255)
    COLOR_STAR_NEAR = (255, 255, 255)
    COLOR_STAR_FAR = (128, 128, 200)

    # Screen
    WIDTH, HEIGHT = 640, 400

    # Player
    PLAYER_X = 150
    PLAYER_WIDTH = 30
    PLAYER_HEIGHT = 20
    GRAVITY = 0.5
    JUMP_VELOCITY = -9
    MAX_JUMPS = 3
    GROUND_Y = HEIGHT - 50

    # Game
    MAX_EPISODE_STEPS = 5000
    STAGES = 3
    STAGE_LENGTH = 4000 # pixels
    OBSTACLE_BASE_SPEED = 3.0
    OBSTACLE_SPEED_INCREASE_INTERVAL = 500
    OBSTACLE_SPEED_INCREASE_AMOUNT = 0.25

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(pygame.font.get_default_font(), 24)
        self.font_small = pygame.font.Font(pygame.font.get_default_font(), 16)

        self.player_pos = None
        self.player_vel_y = None
        self.jumps_left = None
        self.on_ground = None
        self.player_rect = None

        self.obstacles = None
        self.particles = None
        self.stars = None

        self.steps = None
        self.score = None
        self.game_over = None
        self.stage = None
        self.stage_progress = None
        self.obstacle_speed = None
        self.next_obstacle_dist = None
        self.space_pressed_last_frame = None
        
        self.seed = None

        self.reset()
        
        # This can be commented out for performance in production
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.seed = seed
            random.seed(self.seed)
            np.random.seed(self.seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.stage = 1
        self.stage_progress = 0
        self.obstacle_speed = self.OBSTACLE_BASE_SPEED

        self.player_pos = [self.PLAYER_X, self.GROUND_Y]
        self.player_vel_y = 0
        self.jumps_left = self.MAX_JUMPS
        self.on_ground = True
        self.player_rect = pygame.Rect(self.player_pos[0] - self.PLAYER_WIDTH / 2, self.player_pos[1] - self.PLAYER_HEIGHT, self.PLAYER_WIDTH, self.PLAYER_HEIGHT)

        self.obstacles = []
        self.particles = []
        self.next_obstacle_dist = 0
        self._generate_stage_obstacles()

        self.space_pressed_last_frame = False

        if self.stars is None:
            self.stars = [
                (random.randint(0, self.WIDTH), random.randint(0, self.HEIGHT), random.uniform(0.1, 0.5))
                for _ in range(100)
            ]

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0.0

        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        self.stage_progress += self.obstacle_speed

        # --- Update Game Logic ---
        reward += self._update_player(space_held)
        reward += self._update_obstacles()
        self._update_particles()
        self._update_difficulty()

        # --- Calculate Rewards & Score ---
        reward += 0.01  # Reward for surviving
        self.score += 1 # Base score for time

        # --- Handle Collisions & Termination ---
        if self._check_collisions():
            self.game_over = True
            reward = -100.0
            self.score = max(0, self.score - 500)
            self._spawn_particles(self.player_pos, self.COLOR_PLAYER, 50, 8, 'explosion')
            # sfx: player_explosion

        stage_complete, win = self._check_stage_completion()
        if stage_complete:
            reward += 10.0
            self.score += 1000
            self.stage += 1
            if not win:
                self.stage_progress = 0
                self.obstacles = []
                self.next_obstacle_dist = 0
                self._generate_stage_obstacles()
                # sfx: stage_complete
            else:
                self.game_over = True
                reward += 100.0
                self.score += 5000
                # sfx: game_win

        terminated = self.game_over or self.steps >= self.MAX_EPISODE_STEPS or win
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _update_player(self, space_held):
        reward = 0
        space_just_pressed = space_held and not self.space_pressed_last_frame
        
        if space_just_pressed and self.jumps_left > 0:
            self.player_vel_y = self.JUMP_VELOCITY
            self.jumps_left -= 1
            self.on_ground = False
            self._spawn_particles([self.player_pos[0], self.player_pos[1]], self.COLOR_PLAYER, 20, 4, 'jump')
            # sfx: player_jump

            # Check for safe jump
            is_safe = True
            for obs in self.obstacles:
                if self.player_rect.right < obs['rect'].left < self.player_rect.right + 200:
                    is_safe = False
                    break
            if is_safe:
                reward -= 0.2

        self.space_pressed_last_frame = space_held

        self.player_vel_y += self.GRAVITY
        self.player_pos[1] += self.player_vel_y

        if self.player_pos[1] >= self.GROUND_Y:
            self.player_pos[1] = self.GROUND_Y
            self.player_vel_y = 0
            if not self.on_ground:
                self.jumps_left = self.MAX_JUMPS
                self.on_ground = True
                self._spawn_particles([self.player_pos[0], self.player_pos[1]], self.COLOR_STAR_NEAR, 10, 2, 'land')
                # sfx: player_land

        # Keep player within screen bounds (vertically)
        self.player_pos[1] = max(0, self.player_pos[1])
        
        self.player_rect.centerx = int(self.player_pos[0])
        self.player_rect.bottom = int(self.player_pos[1])
        return reward

    def _update_obstacles(self):
        reward = 0
        for obs in self.obstacles:
            obs['rect'].x -= self.obstacle_speed
            
            # Risky jump reward
            if not obs['cleared'] and obs['rect'].right < self.player_rect.left:
                obs['cleared'] = True
                if not self.on_ground:
                    gap = self.player_rect.bottom - obs['rect'].top
                    if -self.PLAYER_HEIGHT < gap < 60: # Narrowly cleared
                        reward += 1.0
                        self.score += 50

        self.obstacles = [obs for obs in self.obstacles if obs['rect'].right > 0]
        return reward

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _update_difficulty(self):
        if self.steps > 0 and self.steps % self.OBSTACLE_SPEED_INCREASE_INTERVAL == 0:
            self.obstacle_speed += self.OBSTACLE_SPEED_INCREASE_AMOUNT

    def _check_collisions(self):
        if self.player_rect.top <= 0:
            return True # Hit ceiling
        for obs in self.obstacles:
            if self.player_rect.colliderect(obs['rect']):
                return True
        return False

    def _check_stage_completion(self):
        if self.stage_progress >= self.STAGE_LENGTH:
            if self.stage >= self.STAGES:
                return True, True  # Stage complete and game won
            return True, False # Stage complete
        return False, False

    def _generate_stage_obstacles(self):
        # Generate obstacles for the entire stage length ahead of time
        current_x = self.WIDTH
        while current_x < self.STAGE_LENGTH:
            gap_x = random.randint(250, 450)
            current_x += gap_x

            gap_y = random.randint(120, 180)
            opening_y = random.randint(80, self.GROUND_Y - gap_y - 20)
            
            # Top obstacle
            height_top = opening_y
            self.obstacles.append({
                'rect': pygame.Rect(current_x, 0, 60, height_top),
                'cleared': False
            })
            # Bottom obstacle
            height_bottom = self.GROUND_Y - (opening_y + gap_y)
            self.obstacles.append({
                'rect': pygame.Rect(current_x, opening_y + gap_y, 60, height_bottom),
                'cleared': False
            })

    def _get_observation(self):
        self._render_game()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "stage": self.stage,
            "jumps_left": self.jumps_left,
        }

    def _render_game(self):
        # Background
        bg_color = list(self.COLOR_BG)
        bg_color[2] = min(255, bg_color[2] + self.stage * 15) # Shift color per stage
        self.screen.fill(tuple(bg_color))

        # Stars (Parallax)
        self._render_stars()

        # Obstacles
        self._render_obstacles()
        
        # Goal line
        self._render_goal()

        # Particles
        self._render_particles()

        # Player
        self._render_player()

        # UI
        self._render_ui()

    def _render_stars(self):
        for i, (x, y, speed) in enumerate(self.stars):
            new_x = (x - (self.stage_progress * speed)) % self.WIDTH
            color = self.COLOR_STAR_NEAR if speed > 0.3 else self.COLOR_STAR_FAR
            size = 2 if speed > 0.3 else 1
            pygame.draw.rect(self.screen, color, (int(new_x), int(y), size, size))

    def _render_obstacles(self):
        for obs in self.obstacles:
            r = obs['rect']
            # Draw glow
            glow_rect = r.inflate(8, 8)
            s = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
            pygame.draw.rect(s, (*self.COLOR_OBSTACLE_GLOW, 50), s.get_rect(), border_radius=5)
            self.screen.blit(s, glow_rect.topleft)
            # Draw main obstacle
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, r, border_radius=3)
            pygame.gfxdraw.rectangle(self.screen, r, (*self.COLOR_OBSTACLE, 180))


    def _render_goal(self):
        goal_x = self.STAGE_LENGTH - self.stage_progress
        if self.WIDTH - 200 < goal_x < self.WIDTH + 20:
            pygame.draw.line(self.screen, self.COLOR_GOAL, (goal_x, 0), (goal_x, self.HEIGHT), 5)

    def _render_particles(self):
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['life'] / p['max_life']))))
            color = (*p['color'], alpha)
            pos = [int(p['pos'][0]), int(p['pos'][1])]
            s = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
            pygame.draw.circle(s, color, (p['size'], p['size']), p['size'])
            self.screen.blit(s, (pos[0] - p['size'], pos[1] - p['size']), special_flags=pygame.BLEND_RGBA_ADD)

    def _render_player(self):
        if self.game_over: return
        
        # Bobbing animation on ground
        bob = math.sin(self.steps * 0.2) * 2 if self.on_ground else 0
        rect = self.player_rect.copy()
        rect.y += bob

        # Create player polygon shape
        p1 = (rect.centerx, rect.top - 5)
        p2 = (rect.right + 5, rect.centery)
        p3 = (rect.centerx, rect.bottom + 5)
        p4 = (rect.left - 5, rect.centery)
        points = [p1, p2, p3, p4]

        # Draw glow
        s = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        pygame.gfxdraw.aapolygon(s, points, (*self.COLOR_PLAYER_GLOW, 80))
        pygame.gfxdraw.filled_polygon(s, points, (*self.COLOR_PLAYER_GLOW, 80))
        self.screen.blit(s, (0, 0))

        # Draw main shape
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Stage
        stage_text = self.font_large.render(f"STAGE: {self.stage}/{self.STAGES}", True, self.COLOR_TEXT)
        self.screen.blit(stage_text, (self.WIDTH - stage_text.get_width() - 10, 10))
        
        # Jumps
        jump_text = self.font_small.render(f"JUMPS: {'● ' * self.jumps_left}{'○ ' * (self.MAX_JUMPS - self.jumps_left)}", True, self.COLOR_PLAYER)
        self.screen.blit(jump_text, (10, 40))

        if self.game_over:
            win_text = "LEVEL COMPLETE" if self.stage > self.STAGES else "GAME OVER"
            end_text_surf = self.font_large.render(win_text, True, self.COLOR_GOAL if self.stage > self.STAGES else self.COLOR_OBSTACLE)
            end_text_rect = end_text_surf.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text_surf, end_text_rect)

    def _spawn_particles(self, pos, color, count, max_speed, p_type):
        for _ in range(count):
            if p_type == 'explosion':
                angle = random.uniform(0, 2 * math.pi)
                speed = random.uniform(1, max_speed)
                vel = [math.cos(angle) * speed, math.sin(angle) * speed]
                life = random.randint(20, 40)
            elif p_type == 'jump':
                vel = [random.uniform(-1, 1), random.uniform(1, max_speed)]
                life = random.randint(10, 20)
            elif p_type == 'land':
                vel = [random.uniform(-max_speed, max_speed), random.uniform(-2, -0.5)]
                life = random.randint(5, 15)
            
            self.particles.append({
                'pos': list(pos),
                'vel': vel,
                'life': life,
                'max_life': life,
                'color': color,
                'size': random.randint(2, 5)
            })

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

# Example of how to run the environment
if __name__ == '__main__':
    # Set this to 'human' to visualize, or 'rgb_array' for headless
    render_mode = "human" 

    if render_mode == "human":
        # In human mode, we need a real display.
        # We will subclass the environment to add a human renderer.
        class HumanGameEnv(GameEnv):
            def __init__(self, render_mode="human"):
                super().__init__(render_mode)
                self.human_screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
                pygame.display.set_caption("Hopping Spaceship")

            def _get_observation(self):
                # First, render everything to the internal surface
                super()._render_game()
                # Then, blit this surface to the display
                if self.render_mode == "human":
                    self.human_screen.blit(self.screen, (0, 0))
                    pygame.display.flip()
                # Return the array as usual
                arr = pygame.surfarray.array3d(self.screen)
                return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

            def close(self):
                super().close()

        env = HumanGameEnv(render_mode="human")
    else:
        env = GameEnv(render_mode="rgb_array")

    obs, info = env.reset()
    done = False
    
    # --- Human Controls ---
    # This block is for playing the game manually.
    # For agent training, you would replace this with your agent's action selection.
    if render_mode == "human":
        total_score = 0
        while not done:
            action = env.action_space.sample() # Start with a random action
            action[0] = 0 # No movement
            action[2] = 0 # No shift
            
            # Poll for pygame events
            space_pressed = False
            should_quit = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    should_quit = True

            if should_quit:
                break

            # Get key presses
            keys = pygame.key.get_pressed()
            if keys[pygame.K_SPACE]:
                space_pressed = True
            
            action[1] = 1 if space_pressed else 0
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_score = info['score']
            
            if done:
                print(f"Game Over! Final Score: {total_score}")
                # Wait a bit before resetting
                pygame.time.wait(2000)
                obs, info = env.reset()
                done = False

            env.clock.tick(30) # Run at 30 FPS
        env.close()

    # --- RL Agent Example ---
    # This block shows a basic random agent loop.
    else:
        for _ in range(1000):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                print(f"Episode finished. Score: {info['score']}, Steps: {info['steps']}")
                obs, info = env.reset()
        env.close()