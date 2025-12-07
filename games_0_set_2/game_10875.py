import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T15:13:54.288174
# Source Brief: brief_00875.md
# Brief Index: 875
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the player manipulates a falling shape
    to create stacks and trigger chain reactions for points against a timer.
    The action space is MultiDiscrete([5, 2, 2]) for simultaneous movement and shape transformation.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Manipulate a falling shape to create stacks of cubes. Trigger chain reactions by dropping a sphere onto a cube stack to score points against the clock."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move the active shape. Press space to transform into a sphere and shift to transform into a cube."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 60
    SHAPE_SIZE = 30
    WIN_SCORE = 500
    GAME_DURATION_SECONDS = 60
    MAX_STEPS = GAME_DURATION_SECONDS * FPS

    # Colors
    COLOR_BG_TOP = (10, 20, 40)
    COLOR_BG_BOTTOM = (40, 60, 100)
    COLOR_CUBE = (60, 150, 255)
    COLOR_SPHERE = (255, 80, 80)
    COLOR_PLAYER_GLOW = (255, 255, 100, 100)  # RGBA for transparency
    COLOR_TEXT = (255, 255, 255)
    
    # Physics & Gameplay
    PLAYER_SPEED = 5
    CUBE_FALL_SPEED = 1.0
    SPHERE_FALL_SPEED = 4.0
    TOTAL_SHAPES = 3
    CHAIN_REACTION_STACK_SIZE = 3

    # --- Gymnasium Core Methods ---

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
        self.font_large = pygame.font.SysFont("sans-serif", 50, bold=True)
        self.font_small = pygame.font.SysFont("sans-serif", 30, bold=True)
        
        # State variables are initialized in reset()
        self.shapes = []
        self.particles = []
        self.score = 0
        self.time_remaining = 0
        self.steps = 0
        self.game_over = False
        self.previous_action = [0, 0, 0]
        
        # self.validate_implementation() # Removed for submission

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.time_remaining = self.GAME_DURATION_SECONDS
        self.game_over = False
        self.previous_action = [0, 0, 0]
        self.shapes.clear()
        self.particles.clear()
        
        # Spawn initial shapes
        self._spawn_shape(is_player=True)
        for _ in range(self.TOTAL_SHAPES - 1):
            self._spawn_shape()
            
        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        
        if self.game_over:
            terminated = True
            return (self._get_observation(), 0, terminated, False, self._get_info())
            
        self.steps += 1
        self.time_remaining -= 1 / self.FPS
        
        reward += self._handle_input(action)
        reward += self._update_physics()
        reward += self._check_chain_reactions()
        self._respawn_shapes_if_needed()
        self._update_particles()
        
        terminated = self._check_termination()
        
        if terminated:
            if self.score >= self.WIN_SCORE:
                reward += 100  # Win bonus
            else:
                reward -= 100  # Loss penalty
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
        
    def close(self):
        pygame.quit()

    # --- Game Logic ---

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        player_shape = self._get_player_shape()
        
        if not player_shape or player_shape['settled']:
            self.previous_action = [movement, int(space_held), int(shift_held)]
            return 0

        # Movement
        if movement == 1: player_shape['pos'].y -= self.PLAYER_SPEED
        elif movement == 2: player_shape['pos'].y += self.PLAYER_SPEED
        elif movement == 3: player_shape['pos'].x -= self.PLAYER_SPEED
        elif movement == 4: player_shape['pos'].x += self.PLAYER_SPEED

        # Transformation on button press (0 -> 1 transition)
        space_pressed = space_held and not self.previous_action[1]
        shift_pressed = shift_held and not self.previous_action[2]

        if space_pressed and player_shape['type'] != 'sphere':
            player_shape['type'] = 'sphere'
            self._create_particles(player_shape['pos'], self.COLOR_SPHERE, 20)
            # sfx: transform_to_sphere.wav
        elif shift_pressed and player_shape['type'] != 'cube':
            player_shape['type'] = 'cube'
            self._create_particles(player_shape['pos'], self.COLOR_CUBE, 20)
            # sfx: transform_to_cube.wav
            
        self.previous_action = [movement, int(space_held), int(shift_held)]
        return 0

    def _update_physics(self):
        reward = 0
        self.shapes.sort(key=lambda s: s['pos'].y, reverse=True)
        
        for shape in self.shapes:
            if shape['settled']:
                continue

            # Apply gravity
            fall_speed = self.SPHERE_FALL_SPEED if shape['type'] == 'sphere' else self.CUBE_FALL_SPEED
            shape['pos'].y += fall_speed
            
            # Boundary checks
            shape['pos'].x = max(0, min(self.SCREEN_WIDTH - self.SHAPE_SIZE, shape['pos'].x))
            shape['pos'].y = max(0, shape['pos'].y)

            shape['rect'].topleft = (int(shape['pos'].x), int(shape['pos'].y))

            # Check for collision with the floor
            if shape['pos'].y >= self.SCREEN_HEIGHT - self.SHAPE_SIZE:
                shape['pos'].y = self.SCREEN_HEIGHT - self.SHAPE_SIZE
                shape['settled'] = True
                # sfx: shape_land_floor.wav
                continue

            # Check for collision with other settled shapes
            for other in self.shapes:
                if other is not shape and other['settled']:
                    if shape['rect'].colliderect(other['rect']):
                        shape['pos'].y = other['rect'].top - self.SHAPE_SIZE
                        shape['settled'] = True
                        shape['pos'].x = other['pos'].x
                        
                        if shape['type'] == 'cube' and other['type'] == 'cube':
                            reward += 0.1
                            # sfx: cube_stack.wav
                        else:
                            # sfx: shape_land_shape.wav
                            pass
                        break
        return reward

    def _check_chain_reactions(self):
        reward = 0
        newly_settled_spheres = [s for s in self.shapes if s['settled'] and s.get('just_settled', True) and s['type'] == 'sphere']
        shapes_to_destroy = set()

        for sphere in newly_settled_spheres:
            shape_below = self._get_shape_directly_below(sphere)
            if shape_below and shape_below['type'] == 'cube':
                stack = [shape_below]
                current = shape_below
                while len(stack) < self.CHAIN_REACTION_STACK_SIZE:
                    next_below = self._get_shape_directly_below(current)
                    if next_below and next_below['type'] == 'cube':
                        stack.append(next_below)
                        current = next_below
                    else:
                        break
                
                if len(stack) >= self.CHAIN_REACTION_STACK_SIZE:
                    self.score += 100
                    reward += 10
                    shapes_to_destroy.add(id(sphere))
                    for cube in stack:
                        shapes_to_destroy.add(id(cube))
                    
                    center_pos = sphere['pos'] + pygame.Vector2(self.SHAPE_SIZE / 2, self.SHAPE_SIZE / 2)
                    self._create_particles(center_pos, self.COLOR_SPHERE, 50, 5)
                    self._create_particles(center_pos, self.COLOR_CUBE, 50, 5)
                    # sfx: chain_reaction_explode.wav

        if shapes_to_destroy:
            self.shapes = [s for s in self.shapes if id(s) not in shapes_to_destroy]
            
        for shape in self.shapes:
            shape['just_settled'] = False
        return reward
        
    def _respawn_shapes_if_needed(self):
        while len(self.shapes) < self.TOTAL_SHAPES:
            self._spawn_shape()
        if not self._get_player_shape():
            self._spawn_shape(is_player=True)

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _check_termination(self):
        if self.score >= self.WIN_SCORE or self.time_remaining <= 0 or self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True
        return False

    # --- Helper & Utility Methods ---

    def _get_player_shape(self):
        for shape in self.shapes:
            if shape['player']:
                return shape
        return None

    def _spawn_shape(self, is_player=False):
        for _ in range(10):
            x = random.randint(0, self.SCREEN_WIDTH - self.SHAPE_SIZE)
            y = random.randint(-self.SHAPE_SIZE * 3, -self.SHAPE_SIZE)
            spawn_rect = pygame.Rect(x, y, self.SHAPE_SIZE, self.SHAPE_SIZE)
            if not any(spawn_rect.colliderect(s['rect']) for s in self.shapes):
                break
        else:
            x, y = random.randint(0, self.SCREEN_WIDTH - self.SHAPE_SIZE), -self.SHAPE_SIZE
        
        shape_type = 'cube' if random.random() < 0.7 else 'sphere'
        new_shape = {
            'pos': pygame.Vector2(x, y), 'type': shape_type, 'player': is_player,
            'settled': False, 'just_settled': True,
            'rect': pygame.Rect(x, y, self.SHAPE_SIZE, self.SHAPE_SIZE)
        }
        self.shapes.append(new_shape)

    def _create_particles(self, pos, color, count, speed_multiplier=1):
        center_pos = pos + pygame.Vector2(self.SHAPE_SIZE / 2, self.SHAPE_SIZE / 2)
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 3) * speed_multiplier
            vel = pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed)
            self.particles.append({
                'pos': center_pos.copy(), 'vel': vel, 'life': random.randint(20, 40),
                'color': color, 'radius': random.randint(2, 5)
            })

    def _get_shape_directly_below(self, shape):
        target_y = shape['rect'].bottom
        candidates = [
            other for other in self.shapes 
            if other is not shape and other['settled'] and
            shape['rect'].left < other['rect'].right and
            shape['rect'].right > other['rect'].left and
            other['rect'].top >= target_y - 5
        ]
        return min(candidates, key=lambda c: c['rect'].top) if candidates else None
        
    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "time_remaining": self.time_remaining}
        
    def _get_observation(self):
        self._render_to_surface()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    # --- Rendering ---

    def _render_to_surface(self):
        for y in range(self.SCREEN_HEIGHT):
            ratio = y / self.SCREEN_HEIGHT
            color = tuple(int(self.COLOR_BG_TOP[i] * (1 - ratio) + self.COLOR_BG_BOTTOM[i] * ratio) for i in range(3))
            pygame.draw.line(self.screen, color, (0, y), (self.SCREEN_WIDTH, y))

        for p in self.particles:
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'].x), int(p['pos'].y), int(p['radius']), p['color'])

        for shape in sorted(self.shapes, key=lambda s: not s['player']): # Draw player on top
            shape['rect'].topleft = (int(shape['pos'].x), int(shape['pos'].y))
            
            if shape['player'] and not shape['settled']:
                glow_rect = shape['rect'].inflate(10, 10)
                s = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
                radius = 8 if shape['type']=='cube' else int(glow_rect.width/2)
                pygame.draw.rect(s, self.COLOR_PLAYER_GLOW, s.get_rect(), border_radius=radius)
                self.screen.blit(s, glow_rect.topleft)

            if shape['type'] == 'cube':
                pygame.draw.rect(self.screen, self.COLOR_CUBE, shape['rect'], border_radius=4)
            else:
                center = (shape['rect'].centerx, shape['rect'].centery)
                radius = self.SHAPE_SIZE // 2
                pygame.gfxdraw.filled_circle(self.screen, center[0], center[1], radius, self.COLOR_SPHERE)
                pygame.gfxdraw.aacircle(self.screen, center[0], center[1], radius, self.COLOR_SPHERE)

        score_text = self.font_small.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        time_text = self.font_small.render(f"Time: {max(0, math.ceil(self.time_remaining))}", True, self.COLOR_TEXT)
        self.screen.blit(time_text, (self.SCREEN_WIDTH - time_text.get_width() - 10, 10))
        
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            msg = "YOU WIN!" if self.score >= self.WIN_SCORE else "TIME'S UP"
            color = (100, 255, 100) if self.score >= self.WIN_SCORE else (255, 100, 100)
            end_text = self.font_large.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

if __name__ == '__main__':
    # This block allows you to run the game and play it with your keyboard.
    # It is not used by the evaluation system.
    os.environ['SDL_VIDEODRIVER'] = 'x11' # Or 'windows', 'mac', etc.
    env = GameEnv()
    obs, info = env.reset()
    
    running = True
    game_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Chain Reaction Puzzle")
    clock = pygame.time.Clock()
    total_reward = 0
    
    while running:
        keys = pygame.key.get_pressed()
        movement = 0
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        action = [movement, keys[pygame.K_SPACE], keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]]
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        game_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            pygame.time.wait(3000)
            obs, info = env.reset()
            total_reward = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        clock.tick(GameEnv.FPS)
        
    env.close()