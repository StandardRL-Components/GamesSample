import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame.gfxdraw
import math

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the player controls three bouncing balls
    to hit 10 targets in a procedurally generated maze within a time limit.
    The game emphasizes visual flair with particle effects and smooth animations.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Launch a trio of bouncing balls into a maze to hit all the targets. "
        "Use power shots to trigger score-multiplying chain reactions before time runs out."
    )
    user_guide = (
        "Controls: Use ←→ or ↑↓ arrow keys to aim. Press space to launch the balls. "
        "Hold shift while launching for a power shot."
    )
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 60 # Visual framerate
    MAX_STEPS = 3600 # 60 seconds * 60 steps/sec

    # Colors
    COLOR_BG = (15, 18, 32)
    COLOR_WALL = (60, 65, 80)
    COLOR_TARGET = (255, 50, 80)
    COLOR_TARGET_HIT = (255, 150, 160)
    COLOR_SPARK = (255, 220, 100)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_UI_ACCENT = (255, 180, 0)
    BALL_COLORS = [(80, 150, 255), (80, 255, 150), (255, 80, 200)]

    # Game Mechanics
    NUM_TARGETS = 10
    BALL_RADIUS = 8
    TARGET_RADIUS = 10
    CHAIN_REACTION_RADIUS = 50
    NORMAL_LAUNCH_SPEED = 7
    BOOSTED_LAUNCH_SPEED = 12
    AIM_ROTATION_SPEED = 0.05
    BALL_COLLISION_SPEED_BONUS = 1.25
    WALL_DAMPENING = 0.95

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 16)

        # --- Game State Initialization ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.targets_hit_count = 0
        self.time_remaining = 0
        self.aim_angle = 0
        self.boost_active = False
        self.prev_space_held = False
        self.balls = []
        self.targets = []
        self.walls = []
        self.particles = []
        self.hit_animations = []
        self.launch_cluster_pos = pygame.Vector2(0, 0)
        
        # This is not called in the final version, but useful for development
        # self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.targets_hit_count = 0
        self.time_remaining = self.MAX_STEPS
        self.aim_angle = self.np_random.uniform(0, 2 * math.pi)
        self.boost_active = False
        self.prev_space_held = False
        self.particles.clear()
        self.hit_animations.clear()

        self._generate_maze_and_place_objects()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        
        self._handle_input(action)
        self._update_game_state()

        # Check for direct hits and trigger chain reactions
        newly_hit_targets = self._check_collisions()
        if newly_hit_targets:
            # sfx: target_direct_hit.wav
            reward += 1.0 * len(newly_hit_targets)
            
            chain_hits = 0
            for target in newly_hit_targets:
                chain_hits += self._trigger_chain_reaction(target['pos'])
            reward += 0.1 * chain_hits

        self.time_remaining -= 1
        self.steps += 1
        
        terminated = (self.targets_hit_count >= self.NUM_TARGETS) or (self.time_remaining <= 0)
        truncated = False

        if terminated and not self.game_over:
            self.game_over = True
            if self.targets_hit_count >= self.NUM_TARGETS:
                reward += 100  # Victory bonus
                # sfx: victory_fanfare.wav
            else:
                reward -= 100  # Failure penalty
                # sfx: failure_buzzer.wav
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # Aiming
        if movement in [1, 3]: # Up or Left
            self.aim_angle -= self.AIM_ROTATION_SPEED
        elif movement in [2, 4]: # Down or Right
            self.aim_angle += self.AIM_ROTATION_SPEED
        self.aim_angle %= (2 * math.pi)

        # Launch Power
        self.boost_active = shift_held

        # Launch Action
        if space_held and not self.prev_space_held:
            self._launch_balls()
        self.prev_space_held = space_held

    def _launch_balls(self):
        # sfx: launch.wav
        speed = self.BOOSTED_LAUNCH_SPEED if self.boost_active else self.NORMAL_LAUNCH_SPEED
        vel = pygame.Vector2(math.cos(self.aim_angle), math.sin(self.aim_angle)) * speed
        
        for ball in self.balls:
            if ball['vel'].length_squared() == 0: # Only launch stationary balls
                ball['vel'] = vel.copy()
                for _ in range(20): # Launch particle burst
                    p_vel = vel.rotate(self.np_random.uniform(-30, 30)) * self.np_random.uniform(0.1, 0.5)
                    self.particles.append(self._create_particle(ball['pos'], p_vel, self.COLOR_SPARK, 20, 3))

    def _update_game_state(self):
        # Update balls
        for i, ball in enumerate(self.balls):
            if ball['vel'].length_squared() > 0:
                ball['pos'] += ball['vel']
                
                # Add trail particles
                if self.steps % 2 == 0:
                    p_pos = ball['pos'] - ball['vel'].normalize() * self.BALL_RADIUS
                    p_vel = -ball['vel'] * 0.1
                    self.particles.append(self._create_particle(p_pos, p_vel, ball['color'], 15, 2))

        # Update particles
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1

        # Update hit animations
        self.hit_animations = [h for h in self.hit_animations if h['life'] > 0]
        for h in self.hit_animations:
            h['life'] -= 1
            h['radius'] += h['expansion_rate']

    def _check_collisions(self):
        newly_hit_targets = []
        # Ball-Wall Collisions
        for ball in self.balls:
            collided = False
            for wall in self.walls:
                if wall.colliderect(pygame.Rect(ball['pos'].x - self.BALL_RADIUS, ball['pos'].y - self.BALL_RADIUS, self.BALL_RADIUS*2, self.BALL_RADIUS*2)):
                    # A simple but effective collision response
                    ball['pos'] -= ball['vel'] # Backtrack
                    # Check horizontal vs vertical collision
                    if wall.width > wall.height: # Horizontal wall
                        ball['vel'].y *= -1
                    else: # Vertical wall
                        ball['vel'].x *= -1
                    ball['vel'] *= self.WALL_DAMPENING
                    collided = True
                    # sfx: bounce_wall.wav
                    break
            if collided:
                for _ in range(5):
                    self.particles.append(self._create_particle(ball['pos'], pygame.Vector2(self.np_random.uniform(-1, 1), self.np_random.uniform(-1, 1)), (200,200,200), 10, 1))

        # Ball-Ball Collisions
        for i in range(len(self.balls)):
            for j in range(i + 1, len(self.balls)):
                b1, b2 = self.balls[i], self.balls[j]
                if (b1['pos'] - b2['pos']).length_squared() == 0: continue
                dist_sq = (b1['pos'] - b2['pos']).length_squared()
                if dist_sq < (self.BALL_RADIUS * 2)**2:
                    # sfx: bounce_ball.wav
                    b1['vel'] *= self.BALL_COLLISION_SPEED_BONUS
                    b2['vel'] *= self.BALL_COLLISION_SPEED_BONUS
                    # Simple separation to prevent sticking
                    overlap = (self.BALL_RADIUS * 2) - math.sqrt(dist_sq)
                    direction = (b1['pos'] - b2['pos']).normalize()
                    b1['pos'] += direction * overlap / 2
                    b2['pos'] -= direction * overlap / 2
        
        # Ball-Target Collisions
        for ball in self.balls:
            if ball['vel'].length_squared() == 0: continue
            for target in self.targets:
                if target['active'] and (ball['pos'] - target['pos']).length_squared() < (self.BALL_RADIUS + self.TARGET_RADIUS)**2:
                    target['active'] = False
                    ball['vel'] *= self.BALL_COLLISION_SPEED_BONUS
                    self.targets_hit_count += 1
                    self.hit_animations.append({'pos': target['pos'], 'radius': self.TARGET_RADIUS, 'life': 20, 'expansion_rate': 2})
                    newly_hit_targets.append(target)
        
        return newly_hit_targets

    def _trigger_chain_reaction(self, origin_pos):
        chain_hits = 0
        to_check = [t for t in self.targets if t['active']]
        
        for target in to_check:
            if (target['pos'] - origin_pos).length_squared() < self.CHAIN_REACTION_RADIUS**2:
                if target['active']:
                    # sfx: chain_reaction_hit.wav
                    target['active'] = False
                    self.targets_hit_count += 1
                    chain_hits += 1
                    self.hit_animations.append({'pos': target['pos'], 'radius': self.TARGET_RADIUS, 'life': 15, 'expansion_rate': 1.5})
                    # Create spark effect between origin and chained target
                    direction = (target['pos'] - origin_pos).normalize()
                    dist = (target['pos'] - origin_pos).length()
                    for i in range(int(dist / 4)):
                        p_pos = origin_pos + direction * i * 4
                        self.particles.append(self._create_particle(p_pos, pygame.Vector2(0,0), self.COLOR_SPARK, 10, 2))
                    
                    # Recursively check for further chains from this new hit
                    chain_hits += self._trigger_chain_reaction(target['pos'])
        return chain_hits

    def _create_particle(self, pos, vel, color, life, size):
        return {'pos': pos.copy(), 'vel': vel.copy(), 'color': color, 'life': life, 'size': size}

    def _generate_maze_and_place_objects(self):
        # Maze Generation (Recursive Backtracking)
        cell_w, cell_h = 40, 40
        cols, rows = self.WIDTH // cell_w, self.HEIGHT // cell_h
        grid = [{'visited': False, 'walls': [True, True, True, True]} for _ in range(cols * rows)] # T, R, B, L
        
        stack = []
        current_idx = self.np_random.integers(0, len(grid))
        grid[current_idx]['visited'] = True
        stack.append(current_idx)

        while stack:
            current_idx = stack.pop()
            cx, cy = current_idx % cols, current_idx // cols
            
            neighbors = []
            # Top
            if cy > 0 and not grid[current_idx - cols]['visited']: neighbors.append('T')
            # Right
            if cx < cols - 1 and not grid[current_idx + 1]['visited']: neighbors.append('R')
            # Bottom
            if cy < rows - 1 and not grid[current_idx + cols]['visited']: neighbors.append('B')
            # Left
            if cx > 0 and not grid[current_idx - 1]['visited']: neighbors.append('L')

            if neighbors:
                stack.append(current_idx)
                direction = self.np_random.choice(neighbors)
                
                if direction == 'T':
                    next_idx = current_idx - cols
                    grid[current_idx]['walls'][0] = False
                    grid[next_idx]['walls'][2] = False
                elif direction == 'R':
                    next_idx = current_idx + 1
                    grid[current_idx]['walls'][1] = False
                    grid[next_idx]['walls'][3] = False
                elif direction == 'B':
                    next_idx = current_idx + cols
                    grid[current_idx]['walls'][2] = False
                    grid[next_idx]['walls'][0] = False
                else: # 'L'
                    next_idx = current_idx - 1
                    grid[current_idx]['walls'][3] = False
                    grid[next_idx]['walls'][1] = False
                
                grid[next_idx]['visited'] = True
                stack.append(next_idx)

        # Create wall rects from grid
        self.walls = []
        for i, cell in enumerate(grid):
            cx, cy = (i % cols) * cell_w, (i // cols) * cell_h
            if cell['walls'][0]: self.walls.append(pygame.Rect(cx, cy, cell_w, 2))
            if cell['walls'][1]: self.walls.append(pygame.Rect(cx + cell_w, cy, 2, cell_h))
            if cell['walls'][2]: self.walls.append(pygame.Rect(cx, cy + cell_h, cell_w, 2))
            if cell['walls'][3]: self.walls.append(pygame.Rect(cx, cy, 2, cell_h))
        
        # Add boundary walls
        self.walls.append(pygame.Rect(0, 0, self.WIDTH, 2))
        self.walls.append(pygame.Rect(0, self.HEIGHT - 2, self.WIDTH, 2))
        self.walls.append(pygame.Rect(0, 0, 2, self.HEIGHT))
        self.walls.append(pygame.Rect(self.WIDTH - 2, 0, 2, self.HEIGHT))

        # Place Objects
        open_spaces = []
        for r in range(rows):
            for c in range(cols):
                open_spaces.append(pygame.Vector2(c * cell_w + cell_w/2, r * cell_h + cell_h/2))
        
        self.np_random.shuffle(open_spaces)

        # Place Balls
        self.launch_cluster_pos = open_spaces.pop()
        self.balls = []
        for i in range(3):
            angle = (2 * math.pi / 3) * i
            offset = pygame.Vector2(math.cos(angle), math.sin(angle)) * (self.BALL_RADIUS * 1.5)
            self.balls.append({
                'pos': self.launch_cluster_pos + offset,
                'vel': pygame.Vector2(0, 0),
                'color': self.BALL_COLORS[i]
            })

        # Place Targets
        self.targets = []
        for _ in range(self.NUM_TARGETS):
            if not open_spaces: break
            pos = open_spaces.pop()
            self.targets.append({'pos': pos, 'active': True})

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Walls
        for wall in self.walls:
            pygame.draw.rect(self.screen, self.COLOR_WALL, wall)

        # Particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / p.get('max_life', p['life'] + 1)))
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'].x), int(p['pos'].y), int(p['size']), (*p['color'], alpha))

        # Hit Animations
        for h in self.hit_animations:
            alpha = int(255 * (h['life'] / 20))
            pygame.gfxdraw.aacircle(self.screen, int(h['pos'].x), int(h['pos'].y), int(h['radius']), (*self.COLOR_TARGET_HIT, alpha))

        # Targets
        for target in self.targets:
            if target['active']:
                pygame.gfxdraw.filled_circle(self.screen, int(target['pos'].x), int(target['pos'].y), self.TARGET_RADIUS, self.COLOR_TARGET)
                pygame.gfxdraw.aacircle(self.screen, int(target['pos'].x), int(target['pos'].y), self.TARGET_RADIUS, (255,255,255))
        
        # Aiming indicator if balls are stationary
        are_balls_still = all(b['vel'].length_squared() == 0 for b in self.balls)
        if are_balls_still:
            start_pos = self.launch_cluster_pos
            end_pos = start_pos + pygame.Vector2(math.cos(self.aim_angle), math.sin(self.aim_angle)) * 40
            pygame.draw.line(self.screen, self.COLOR_UI_ACCENT, start_pos, end_pos, 2)
            pygame.gfxdraw.filled_circle(self.screen, int(end_pos.x), int(end_pos.y), 3, self.COLOR_UI_ACCENT)

        # Balls
        for ball in self.balls:
            pos_int = (int(ball['pos'].x), int(ball['pos'].y))
            # Glow effect
            glow_radius = int(self.BALL_RADIUS * 1.5)
            glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, (*ball['color'], 50), (glow_radius, glow_radius), glow_radius)
            self.screen.blit(glow_surf, (pos_int[0] - glow_radius, pos_int[1] - glow_radius))

            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], self.BALL_RADIUS, ball['color'])
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], self.BALL_RADIUS, (255,255,255))

    def _render_ui(self):
        # Time
        time_text = f"TIME: {self.time_remaining / self.FPS:.1f}"
        time_surf = self.font_main.render(time_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(time_surf, (self.WIDTH - time_surf.get_width() - 10, 10))
        
        # Targets
        target_text = f"TARGETS: {self.targets_hit_count}/{self.NUM_TARGETS}"
        target_surf = self.font_main.render(target_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(target_surf, (self.WIDTH - target_surf.get_width() - 10, 35))

        # Boost indicator
        if self.boost_active:
            boost_text = "POWER BOOST"
            boost_surf = self.font_small.render(boost_text, True, self.COLOR_UI_ACCENT)
            self.screen.blit(boost_surf, (self.WIDTH / 2 - boost_surf.get_width() / 2, self.HEIGHT - 30))

        # Game Over Text
        if self.game_over:
            if self.targets_hit_count >= self.NUM_TARGETS:
                end_text = "VICTORY!"
                end_color = self.COLOR_UI_ACCENT
            else:
                end_text = "TIME UP!"
                end_color = self.COLOR_TARGET
            end_surf = pygame.font.SysFont("Consolas", 50, bold=True).render(end_text, True, end_color)
            self.screen.blit(end_surf, (self.WIDTH / 2 - end_surf.get_width() / 2, self.HEIGHT / 2 - end_surf.get_height() / 2))


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "targets_hit": self.targets_hit_count,
            "time_remaining": self.time_remaining
        }
        
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
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
        assert trunc is False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # To do so, you must unset the headless environment variable
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Chain Reaction Maze")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement = 0 # no-op
        space = 0
        shift = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        if keys[pygame.K_DOWN]: movement = 2
        if keys[pygame.K_LEFT]: movement = 3
        if keys[pygame.K_RIGHT]: movement = 4
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
            
        action = [movement, space, shift]
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Display the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}")
            print("Press 'R' to reset.")
            
        clock.tick(GameEnv.FPS)

    env.close()